import os
import re

from typing import List, Tuple, Any, Optional, Set
from datetime import datetime

import toml
import jsonlines
import spacy
import torch

import coref.const as const
from coref.utils import add_dummy, GraphNode
from coref.config import Config

nlp = spacy.blank("en")


def _sentids_to_sentstarts(
    sent_ids: List[int]
) -> List[int]:
    """
    Convert sentence id per token to sentence start indicators.
    """
    sent_starts = [1]
    for i in range(1, len(sent_ids)):
        start = int(sent_ids[i] != sent_ids[i - 1])
        sent_starts.append(start)
    return sent_starts


def _convert_to_spacy_doc(
    doc: const.Doc
) -> spacy.tokens.Doc:
    """
    Just converts sentence-ids to sentence starts basically.
    """
    sent_starts = _sentids_to_sentstarts(doc['sent_id'])
    return spacy.tokens.Doc(vocab=nlp.vocab,
                            words=doc['cased_words'],
                            sent_starts=sent_starts)


def _cluster_ids(
    doc: const.Doc, device
) -> torch.Tensor:
    """
    Returns a torch.Tensor of shape [n_word], containing cluster
    indices for each word. Non-coreferent words have cluster id of zero.
    """
    word2cluster = {word_i: i
                    for i, cluster in enumerate(doc["word_clusters"], start=1)
                    for word_i in cluster}

    return torch.tensor(
        [word2cluster.get(word_i, 0)
         for word_i in range(len(doc["cased_words"]))],
        device=device
    )


def _get_ground_truth(
    cluster_ids: torch.Tensor,
    top_indices: torch.Tensor,
    valid_pair_map: torch.Tensor
) -> torch.Tensor:
    """
    cluster_ids: tensor of shape [n_words], containing cluster indices
        for each word. Non-gold words have cluster id of zero.
    top_indices: tensor of shape [n_words, n_ants],
        indices of antecedents of each word
    valid_pair_map: boolean tensor of shape [n_words, n_ants],
        whether for pair at [i, j] (i-th word and j-th word)
        j < i is True

    Returns:
        tensor of shape [n_words, n_ants + 1] (dummy added),
            containing 1 at position [i, j] if i-th and j-th words corefer.
    """
    y = cluster_ids[top_indices] * valid_pair_map  # [n_words, n_ants]
    y[y == 0] = -1                                 # -1 for non-gold words
    y = add_dummy(y)                         # [n_words, n_cands + 1]
    y = (y == cluster_ids.unsqueeze(1))            # True if coreferent
    # For all rows with no gold antecedents setting dummy to True
    y[y.sum(dim=1) == 0, 0] = True
    return y.to(torch.float)


def _clusterize(
        doc: const.Doc,
        scores: torch.Tensor,
        top_indices: torch.Tensor
):
    antecedents = scores.argmax(dim=1) - 1
    not_dummy = antecedents >= 0
    coref_span_heads = torch.arange(0, len(scores))[not_dummy]
    antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]

    nodes = [GraphNode(i) for i in range(len(doc["cased_words"]))]
    for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
        nodes[i].link(nodes[j])
        assert nodes[i] is not nodes[j]

    clusters = []
    for node in nodes:
        if len(node.links) > 0 and not node.visited:
            cluster = []
            stack = [node]
            while stack:
                current_node = stack.pop()
                current_node.visited = True
                cluster.append(current_node.id)
                stack.extend(link for link in current_node.links if not link.visited)
            assert len(cluster) > 1
            clusters.append(sorted(cluster))
    return sorted(clusters)


# XXX
def _tokenize_docs(path: str) -> List[const.Doc]:
    """
    Since the spacy-transformers are taking over the only
    thing its doing at the moment is to cast a list of 
    lists to a list of tuples, don't know why yet :D. 
    """
    print(f"Tokenizing documents at {path}...", flush=True)
    out: List[Doc] = []
    with jsonlines.open(path, mode="r") as data_f:
        for doc in data_f:
            doc["span_clusters"] = [[tuple(mention) for mention in cluster]
                               for cluster in doc["span_clusters"]]
            out.append(doc)
    print("Tokenization OK", flush=True)
    return out


def _load_config(
    config_path: str,
    section: str
) -> Config:
    config = toml.load(config_path)
    default_section = config["DEFAULT"]
    current_section = config[section]
    unknown_keys = (set(current_section.keys())
                    - set(default_section.keys()))
    if unknown_keys:
        raise ValueError(f"Unexpected config keys: {unknown_keys}")
    return Config(section, **{**default_section, **current_section})


def save_state(model, optimizer):
    """ Saves trainable models as state dicts. """
    to_save = [(key, value) for key, value in model.trainable.items() if key != "bert"]
    to_save.extend([('optimizer', optimizer)])
    time = datetime.strftime(datetime.now(), "%Y.%m.%d_%H.%M")
    path = os.path.join(model.config.data_dir,
                        f"{model.config.section}"
                        f"_(e{model.epochs_trained}_{time}).pt")
    savedict = {name: module.state_dict() for name, module in to_save}
    savedict["epochs_trained"] = model.epochs_trained  # type: ignore
    torch.save(savedict, path)


def load_state(
    model,
    optimizer: Optional = None,
    path: Optional[str] = None,
    ignore: Optional[Set[str]] = None,
    map_location: Optional[str] = None,
    noexception: bool = False
) -> None:
    """
    Loads pretrained weights of modules saved in a file located at path.
    If path is None, the last saved model with current configuration
    in data_dir is loaded.
    Assumes files are named like {configuration}_(e{epoch}_{time})*.pt.
    """
    if path is None:
        pattern = rf"{model.config.section}_\(e(\d+)_[^()]*\).*\.pt"
        files = []
        for f in os.listdir(model.config.data_dir):
            match_obj = re.match(pattern, f)
            if match_obj:
                files.append((int(match_obj.group(1)), f))
        if not files:
            if noexception:
                print("No weights have been loaded", flush=True)
                return
            raise OSError(f"No weights found in {self.config.data_dir}!")
        _, path = sorted(files)[-1]
        path = os.path.join(model.config.data_dir, path)

    if map_location is None:
        map_location = self.config.device
    print(f"Loading from {path}...")
    state_dicts = torch.load(path, map_location=map_location)
    model.epochs_trained = state_dicts.pop("epochs_trained", 0)
    for key, state_dict in state_dicts.items():
        if not ignore or key not in ignore:
            if key.endswith("optimizer"):
                optimizers[key].load_state_dict(state_dict)
            model.trainable[key].load_state_dict(state_dict)
    return model, optimizer
