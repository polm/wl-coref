import spacy
import torch

from typing import Tuple, List
from thinc.types import Floats2d, Ints1d, Ints2d
from thinc.util import xp2torch, torch2xp, is_torch_array
from coref import const
from thinc.api import PyTorchWrapper, Model, reduce_mean, chain, tuplify
from thinc.api import ArgsKwargs, torch2xp, xp2torch
from spacy_transformers.architectures import transformer_tok2vec_v3
from spacy_transformers.span_getters import configure_strided_spans

from coref.coref_model import CorefScorer
from coref.span_predictor import SpanPredictor


# XXX this global nlp feels sketchy
nlp = spacy.blank("en")


class GraphNode:
    def __init__(self, node_id: int):
        self.id = node_id
        self.links: Set[GraphNode] = set()
        self.visited = False

    def link(self, another: "GraphNode"):
        self.links.add(another)
        another.links.add(self)

    def __repr__(self) -> str:
        return str(self.id)


def convert_coref_scorer_inputs(
    model: Model,
    X: Tuple[const.Doc, Floats2d, Ints1d], 
    is_train: bool
):
    doc = X[0]
    word_features = xp2torch(X[1], requires_grad=False)
    cluster_ids = xp2torch(X[2], requires_grad=False)
    return ArgsKwargs(args=(doc, word_features, cluster_ids), kwargs={}), lambda dX: []


def convert_coref_scorer_outputs(
    model: Model,
    inputs_outputs,
    is_train: bool
):
    _, outputs = inputs_outputs
    scores, indices = outputs
    
    def convert_for_torch_backward(dY: Floats2d) -> ArgsKwargs:
        dY_t = xp2torch(dY)
        return ArgsKwargs(
            args=([scores],),
            kwargs={"grad_tensors": [dY_t]},
        )

    scores_xp = torch2xp(scores)
    indices_xp = torch2xp(indices)
    return (scores_xp, indices_xp), convert_for_torch_backward


def convert_span_predictor_inputs(
    model: Model,
    X: Tuple[const.Doc, Floats2d, Ints1d], 
    is_train: bool
):
    doc = X[0]
    word_features = xp2torch(X[1], requires_grad=False)
    head_ids = xp2torch(X[2], requires_grad=False)
    return ArgsKwargs(args=(doc, word_features, head_ids), kwargs={}), lambda dX: []


def spaCyRoBERTa(
) -> Model[spacy.tokens.Doc, List[Floats2d]]:
    """Configures and returns RoBERTa from spacy-transformers."""
    return transformer_tok2vec_v3(
        name='roberta-large',
        get_spans=configure_strided_spans(400, 350),
        tokenizer_config={'add_prefix_space': True},
        pooling=reduce_mean()
    )


def cluster_ids(
) -> Model[const.Doc, Ints1d]:
    """
    Gets cluster_ids for each word in the Doc.
    """
    return Model(
        'cluster_ids',
        forward_cluster_ids,
    )


def doc2doc(
) -> Model[const.Doc, spacy.tokens.Doc]:
    """Convert Doc object from this library to spacy.tokens.Doc."""
    return Model('doc2doc', forward_doc2doc)


def doc2spaninfo(
) -> Model[const.Doc, Tuple[Ints1d, Ints1d, Ints1d]]:
    """Convert Doc to input to the SpanResolver model."""
    return Model(
        'prepare_for_spanresolver',
        forward_doc2spaninfo
    )


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


def forward_cluster_ids(
    model: Model[const.Doc, Ints1d],
    doc: const.Doc,
    is_train: bool
) -> torch.Tensor:
    word2cluster = {word_i: i
                    for i, cluster in enumerate(doc["word_clusters"], start=1)
                    for word_i in cluster}
    clusters_list= [word2cluster.get(word_i, 0)
                    for word_i in range(len(doc["cased_words"]))]

    def backprop(dY):
        return []

    return model.ops.xp.asarray(clusters_list), backprop


def forward_doc2doc(
    model: Model[const.Doc, spacy.tokens.Doc],
    doc: const.Doc,
    is_train: bool
) -> List[spacy.tokens.Doc]:
    sent_starts = _sentids_to_sentstarts(doc['sent_id'])

    def backprop(dY):
        return []

    spacy_doc = spacy.tokens.Doc(vocab=nlp.vocab,
                                 words=doc['cased_words'],
                                 sent_starts=sent_starts)
    return [spacy_doc], backprop


def forward_doc2spaninfo(
    model: Model[const.Doc,
                 Tuple[Ints1d, Ints1d, Ints1d]],
    doc: const.Doc,
    is_train: bool
) -> Tuple[Ints1d, Ints1d, Ints1d]:
    head2span = sorted(doc["head2span"])

    def backprop(dY):
        return []

    if not head2span:

        return (model.ops.xp.asarray([]),
                model.ops.xp.asarray([]),
                model.ops.xp.asarray([])
                ), backprop

    heads, starts, ends = zip(*head2span)
    heads = model.ops.xp.asarray(heads)
    starts = model.ops.xp.asarray(starts)
    ends = model.ops.xp.asarray(ends) - 1


    return (heads, starts, ends), backprop


def doc2inputs(
) -> Model[const.Doc, Tuple[Floats2d, Ints1d]]:
    """
    Create pipeline that goes from Doc to RoBERTa
    features and cluster_ids.
    """
    encoder = chain(doc2doc(), spaCyRoBERTa())
    return tuplify(encoder, cluster_ids())


def configure_pytorch_modules(config):
    """
    Initializes CorefScorer and SpanPredictor Pytorch modules.
    """
    coref_scorer = PyTorchWrapper(
        CorefScorer(
            config.device,
            config.embedding_size,
            config.hidden_size,
            config.n_hidden_layers,
            config.dropout_rate,
            config.rough_k,
            config.a_scoring_batch_size
        ),
        convert_inputs=convert_coref_scorer_inputs,
        convert_outputs=convert_coref_scorer_outputs
    )
    span_predictor = PyTorchWrapper(
        SpanPredictor(
            1024,
            config.sp_embedding_size,
            config.device
        ),
        convert_inputs=convert_span_predictor_inputs
    )
    coref_scorer.initialize()
    span_predictor.initialize()
    return coref_scorer, span_predictor


def save_state(model, span_predictor, config):
    """Serialize CorefScorer and SpanPredictor."""
    time = datetime.strftime(datetime.now(), "%Y.%m.%d_%H.%M")
    span_path = os.path.join(config.data_dir,
                        f"span-{config.section}"
                        f"_(e{model.attrs['epochs_trained']}_{time}).pt")
    coref_path = os.path.join(config.data_dir,
                        f"coref-{config.section}"
                        f"_(e{model.attrs['epochs_trained']}_{time}).pt")
    model.to_disk(coref_path)
    span_predictor.to_disk(span_path)


def load_state(
    model,
    span_predictor,
    config,
    coref_path,
    span_path,
):
    """Deserialize CorefScorer and SpanPredictor."""
    model.from_disk(coref_path)
    span_predictor.from_disk(span_path)
    return model, span_predictor


def predict_span_clusters(span_predictor: Model,
                          doc: const.Doc,
                          words: Floats2d,
                          clusters: List[Ints1d]):
    """
    Predicts span clusters based on the word clusters.

    Args:
        doc (Doc): the document data
        words (torch.Tensor): [n_words, emb_size] matrix containing
            embeddings for each of the words in the text
        clusters (List[List[int]]): a list of clusters where each cluster
            is a list of word indices

    Returns:
        List[List[Span]]: span clusters
    """
    if not clusters:
        return []

    xp = span_predictor.ops.xp
    heads_ids = xp.asarray(sorted(i for cluster in clusters for i in cluster))
    scores = span_predictor.predict((doc, words, heads_ids))
    starts = scores[:, :, 0].argmax(axis=1).tolist()
    ends = (scores[:, :, 1].argmax(axis=1) + 1).tolist()

    head2span = {
        head: (start, end)
        for head, start, end in zip(heads_ids.tolist(), starts, ends)
    }

    return [[head2span[head] for head in cluster]
            for cluster in clusters]


def _clusterize(
        model,
        scores: Floats2d,
        top_indices: Ints2d
):
    xp = model.ops.xp
    antecedents = scores.argmax(axis=1) - 1
    not_dummy = antecedents >= 0
    coref_span_heads = xp.arange(0, len(scores))[not_dummy]
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
