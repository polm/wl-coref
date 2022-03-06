from typing import Tuple

import argparse
import tqdm
from contextlib import contextmanager
import datetime
import random
import time

import numpy as np  # type: ignore
import torch        # type: ignore

from coref.cluster_checker import ClusterChecker
from coval import Evaluator, get_cluster_info, b_cubed, muc, ceafe, lea
from thinc.api import require_gpu
from thinc.api import Adam
from coref.utils import _load_config
from coref.thinc_funcs import configure_pytorch_modules, doc2tensors
from coref.thinc_funcs import spaCyRoBERTa
from coref.thinc_funcs import _clusterize, predict_span_clusters
from coref.thinc_funcs import load_state, save_state
from coref.thinc_funcs import get_head2span
from coref.thinc_loss import coref_loss, span_loss
from convert_to_spacy import load_spacy_data


def train(
    config,
    model,
    span_predictor,
):
    """
    Trains all the trainable blocks in the model using the config provided.
    """
    docs = load_spacy_data('train.spacy')
    optimizer = Adam(config.learning_rate)
    docs_ids = list(range(len(docs)))
    best_val_score = 0
    encoder = spaCyRoBERTa()
    encoder.initialize()


    for epoch in range(model.attrs['epochs_trained'], config.train_epochs):
        running_c_loss = 0.0
        running_s_loss = 0.0
        random.shuffle(docs_ids)
        pbar = tqdm.tqdm(docs_ids, unit="docs", ncols=0)
        for doc_id in pbar:
            doc = docs[doc_id]
            # Get data for CorefScorer
            sent_ids, cluster_ids, heads, starts, ends = doc2tensors(
                model.ops.xp,
                doc
            )
            word_features, _ = encoder([doc], False)
            # Run CorefScorer
            (coref_scores, top_indices), backprop = model.begin_update(
                word_features[0]
            )
            # Compute coref loss
            c_loss, c_grads = coref_loss(
                model,
                cluster_ids,
                coref_scores,
                top_indices
            )
            # Update CorefScorer
            backprop(c_grads)
            model.finish_update(optimizer)
            # Get data for SpanPredictor
            # (heads, starts, ends), _ = span_provider(doc, False)
            if starts.size and ends.size:
                span_scores, backprop_span = span_predictor.begin_update(
                    (
                        sent_ids,
                        word_features[0],
                        heads
                    )
                )
                s_loss, s_grads = span_loss(
                    span_predictor,
                    span_scores,
                    starts,
                    ends
                )
                backprop_span(s_grads)
                span_predictor.finish_update(optimizer)
                del span_scores
            else:
                s_loss = 0

            running_c_loss += c_loss
            running_s_loss += s_loss

            del coref_scores
            del top_indices

            pbar.set_description(
                f"Epoch {epoch + 1}:"
                #f" {doc._.document_id:26}"
                f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
            )

        model.attrs['epochs_trained'] += 1
        val_score = evaluate(config, model, span_predictor)
        if val_score > best_val_score:
            best_val_score = val_score
            print("New best {}".format(best_val_score))
            save_state(model, span_predictor, optimizer, config)

def get_gold_word_clusters(doc):
    """Return world-level clusters as a list of lists of token indices."""
    clusters = []
    for key, sg in doc.spans.items():
        if not key.startswith("coref_word_clusters_"):
            continue

        # it's a group of spans, but each span is just one token
        cluster = [span[0].i for span in sg]
        clusters.append(cluster)
    return clusters

def get_gold_clusters(doc):
    """Return clusters as list of span indices."""
    clusters = []
    for key, sg in doc.spans.items():
        if not key.startswith("coref_clusters_"):
            continue

        cluster = [(span.start, span.end) for span in sg]
        if len(cluster) == 0:
            print("===== !!! empty cluster !!!! =====")
        clusters.append(cluster)
    return clusters

def get_head2span(doc):
    """Return head to span mapping.

    Head to span mapping is a list up tuples of (head token index, span start,
    span end).

    This assumes that the order of word clusters and span clusters in the doc,
    and the order of spans in those clusters, are aligned.
    """
    # XXX This doesn't actually work because of alignment issues.
    # For now just assume we don't have this.
    head_groups = [doc.spans[sk] for sk in doc.spans if sk.startswith("coref_word_clusters")]
    heads = []
    for group in head_groups:
        for span in group:
            # note this are all one-token spans
            heads.append(span[0].i)

    spans = []
    groups = [doc.spans[sk] for sk in doc.spans if sk.startswith("coref_clusters")]
    for group in groups:
        for ss in group:
            spans.append( (ss.start, ss.end) )

    out = []
    for head, (start, end) in zip(heads, spans):
        out.append( (head, start, end) )
    return out

@torch.no_grad()
def evaluate(
    config,
    model,
    span_predictor,
    data_split: str = "dev",
    word_level_conll: bool = False
) -> Tuple[float, Tuple[float, float, float]]:

    encoder = spaCyRoBERTa()
    encoder.initialize()
    docs = load_spacy_data('dev.spacy')
    n_docs = len(docs)
    running_loss = 0.0
    w_checker = ClusterChecker()
    s_checker = ClusterChecker()
    muc_evaluator = Evaluator(muc)
    bcubed_evaluator = Evaluator(b_cubed)
    ceafe_evaluator = Evaluator(ceafe)
    lea_evaluator = Evaluator(lea)
    s_correct = 0
    s_total = 0
    muc_score = 0.
    bcubed_score = 0.
    ceafe_score = 0.
    lea_score = 0.

    pbar = tqdm.tqdm(docs, unit="docs", ncols=0)
    for i, doc in enumerate(pbar):
        doc = docs[i]
        sent_ids, cluster_ids, heads, starts, ends = doc2tensors(
            model.ops.xp,
            doc
        )
        word_features, _ = encoder([doc], False)
        # Get data for SpanPredictor
        # Run CorefScorer
        coref_scores, top_indices = model.predict(word_features[0])
        # Compute coreference loss
        c_loss, c_grads = coref_loss(
            model,
            cluster_ids,
            coref_scores,
            top_indices
        )
        word_clusters = _clusterize(
             span_predictor,
             coref_scores,
             top_indices
         )
        running_loss += c_loss
        if starts.size and ends.size:
            span_scores = span_predictor.predict(
                (
                    sent_ids,
                    word_features[0],
                    heads
                )
            )
            span_clusters = predict_span_clusters(
                span_predictor,
                sent_ids,
                word_features[0],
                word_clusters
            )
            pred_starts = span_scores[:, :, 0].argmax(axis=1)
            pred_ends = span_scores[:, :, 1].argmax(axis=1)
            s_correct += ((starts == pred_starts) * (ends == pred_ends)).sum()
            s_total += len(pred_starts)

        gold_word_clusters = get_gold_word_clusters(doc)
        w_checker.add_predictions(gold_word_clusters, word_clusters)
        w_lea = w_checker.total_lea
        gold_clusters = get_gold_clusters(doc)
        s_checker.add_predictions(gold_clusters, span_clusters)
        s_lea = s_checker.total_lea

        cluster_info = get_cluster_info(
            span_clusters,
            gold_clusters
        )
        muc_evaluator.update(cluster_info)
        bcubed_evaluator.update(cluster_info)
        ceafe_evaluator.update(cluster_info)
        lea_evaluator.update(cluster_info)
        muc_score += muc_evaluator.get_f1()
        bcubed_score += bcubed_evaluator.get_f1()
        ceafe_score += ceafe_evaluator.get_f1()
        lea_score += lea_evaluator.get_f1()

        pbar.set_description(
            f"{data_split}:"
            f" | WL: "
            f" loss: {running_loss / (pbar.n + 1):<.5f},"
            f" f1: {w_lea[0]:.5f},"
            f" p: {w_lea[1]:.5f},"
            f" r: {w_lea[2]:<.5f}"
            f" | SL: "
            f" sa: {s_correct / s_total:<.5f},"
            f" f1: {s_lea[0]:.5f},"
            f" p: {s_lea[1]:.5f},"
            f" r: {s_lea[2]:<.5f}"
        )
    print("LEA", s_lea[0])
    print("Paul LEA", lea_score/n_docs)
    print("MUC", muc_score/n_docs)
    print("CEAFE", ceafe_score/n_docs)
    print("BCUB", bcubed_score/n_docs)
    eval_score = w_lea[0] + s_lea[0]
    return eval_score


@contextmanager
def output_running_time():
    """ Prints the time elapsed in the context """
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """ Seed random number generators to get reproducible results """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)           # type: ignore
    torch.backends.cudnn.deterministic = True   # type: ignore
    torch.backends.cudnn.benchmark = False      # type: ignore


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=("train", "eval"))
    argparser.add_argument("experiment")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--data-split", choices=("train", "dev", "test"),
                           default="test",
                           help="Data split to be used for evaluation."
                                " Defaults to 'test'."
                                " Ignored in 'train' mode.")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--coref-weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--span-weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--word-level", action="store_true",
                           help="If set, output word-level conll-formatted"
                                " files in evaluation modes. Ignored in"
                                " 'train' mode.")
    args = argparser.parse_args()

    if args.batch_size:
        config.a_scoring_batch_size = args.batch_size

    require_gpu()
    seed(2020)
    config = _load_config(args.config_file, args.experiment)
    model, span_predictor = configure_pytorch_modules(config)
    
    if args.mode == "train":
        model.attrs['epochs_trained'] = 0
        with output_running_time():
            train(config, model, span_predictor)
    else:
        model, span_predictor = load_state(
            model,
            span_predictor,
            config,
            args.coref_weights,
            args.span_weights,
        )
        evaluate(config,
                 model,
                 span_predictor,
                 data_split=args.data_split,
                 word_level_conll=args.word_level)
