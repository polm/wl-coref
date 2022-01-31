import os
import re
import pickle


import toml
import jsonlines
import spacy
import torch
import coref.const as const

from typing import List, Tuple, Any, Optional, Set
from datetime import datetime

from coref.utils import add_dummy, GraphNode
from coref.config import Config


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
        scores: torch.Tensor,
        top_indices: torch.Tensor
):
    antecedents = scores.argmax(dim=1) - 1
    not_dummy = antecedents >= 0
    coref_span_heads = torch.arange(0, len(scores))[not_dummy]
    antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]
    n_words = scores.shape[0]
    nodes = [GraphNode(i) for i in range(n_words)]
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


# XXX this is completely pointless at this point
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


# XXX will go away and be handled by thinc
def save_state(model, span_predictor, optimizer):
    """ Saves trainable models as state dicts. """
    time = datetime.strftime(datetime.now(), "%Y.%m.%d_%H.%M")
    path = os.path.join(model.config.data_dir,
                        f"{model.config.section}"
                        f"_(e{model.epochs_trained}_{time}).pt")
    savedict = {'model': model.state_dict(),
                'span_predictor': span_predictor.state_dict(),
                'epochs_trained': model.epochs_trained,
                'optimizer': optimizer}
    torch.save(savedict, path)


# XXX will go away for and handled by thinc
def load_state(
    model,
    span_predictor,
    optimizer: Optional = None,
    path: Optional[str] = None,
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
            raise OSError(f"No weights found in {model.config.data_dir}!")
        _, path = sorted(files)[-1]
        path = os.path.join(model.config.data_dir, path)

    if map_location is None:
        map_location = self.config.device
    print(f"Loading from {path}...")
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    span_predictor.load_state_dict(checkpoint['span_predictor'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    model.epochs_trained = checkpoint.get("epochs_trained", 0)
    return model, span_predictor, optimizer


# XXX will go away and replaced with a spaCy loader
def get_docs(path: str,
              bert_model: str) -> List[const.Doc]:
    """
    Either grabs jsonlines.pickle or creates it.
    """
    basename = os.path.basename(path)
    model_name = bert_model.replace("/", "_")
    # model_name = self.config.bert_model.replace("/", "_")
    cache_filename = f"{model_name}_{basename}.pickle"
    if os.path.exists(cache_filename):
        with open(cache_filename, mode="rb") as cache_f:
            docs = pickle.load(cache_f)
    else:
        docs = _tokenize_docs(path)
        with open(cache_filename, mode="wb") as cache_f:
            pickle.dump(docs, cache_f)
    return docs
