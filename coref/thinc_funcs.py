import spacy
import torch

from typing import Tuple, List
from thinc.types import Floats2d, Ints1d

from coref import const
from thinc.api import Model, reduce_first, chain, tuplify

from spacy_transformers.architectures import transformer_tok2vec_v3
from spacy_transformers.span_getters import configure_strided_spans


# XXX this global nlp feels sketchy
nlp = spacy.blank("en")


def spaCyRoBERTa(
) -> Model[spacy.tokens.Doc, List[Floats2d]]:
    """Configures and returns RoBERTa from spacy-transformers."""
    return transformer_tok2vec_v3(
        name='roberta-large',
        get_spans=configure_strided_spans(400, 350),
        tokenizer_config={'add_prefix_space': True},
        pooling=reduce_first()
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
        empty = model.ops.xp.asarray([])
        return ([], [], []), backprop
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
