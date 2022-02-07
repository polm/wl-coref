from typing import Tuple

import argparse
import tqdm
from contextlib import contextmanager
import datetime
import random
import sys
import time

import numpy as np  # type: ignore
import torch        # type: ignore

from coref import conll
from coref.cluster_checker import ClusterChecker
from coref.const import Doc
from thinc.api import require_gpu
from thinc.api import PyTorchWrapper
from thinc.api import Adam as Tadam
from coref.spacy_util import get_docs, _load_config
from coref.thinc_funcs import doc2inputs, doc2spaninfo, configure_pytorch_modules
from coref.thinc_funcs import _clusterize, predict_span_clusters
from coref.thinc_funcs import load_state, save_state
from coref.thinc_loss import coref_loss, span_loss


def train(
    config,
    model: CorefScorer,
    span_predictor: SpanPredictor,
):
    """
    Trains all the trainable blocks in the model using the config provided.
    """
    docs = list(get_docs(config.train_data,
                         config.bert_model))
    data_provider = doc2inputs()
    span_provider = doc2spaninfo()
    coref_optimizer = Tadam(config.learning_rate)
    span_optimizer = Tadam(config.learning_rate)
    data_provider.initialize()
    span_provider.initialize()
    n_docs = len(docs)
    docs_ids = list(range(len(docs)))
    best_val_score = 0

    for epoch in range(model.attrs['epochs_trained'], config.train_epochs):
        running_c_loss = 0.0
        running_s_loss = 0.0
        random.shuffle(docs_ids)
        pbar = tqdm.tqdm(docs_ids, unit="docs", ncols=0)
        for doc_id in pbar:
            doc = docs[doc_id]
            # Get data for CorefScorer
            (word_features, cluster_ids), _ = data_provider(doc, False)
            # Run CorefScorer
            (coref_scores, top_indices), backprop = model.begin_update(
                (
                    doc,
                    word_features[0],
                    cluster_ids
                )
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
            model.finish_update(coref_optimizer)
            # Get data for SpanPredictor
            (heads, starts, ends), _ = span_provider(doc, False)
            if starts.size and ends.size:
                span_scores, backprop_span = span_predictor.begin_update(
                    (
                        doc,
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
                span_predictor.finish_update(span_optimizer)
                del span_scores
            else:
                s_loss = 0

            running_c_loss += c_loss
            running_s_loss += s_loss

            del coref_scores
            del top_indices


            pbar.set_description(
                f"Epoch {epoch + 1}:"
                f" {doc['document_id']:26}"
                f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
            )

        model.attrs['epochs_trained'] += 1
        val_score = evaluate(config, model, span_predictor)
        if val_score > best_val_score:
            best_val_score = val_score
            print("New best {}".format(best_val_score))
            save_state(model, span_predictor, config)


@torch.no_grad()
def evaluate(
    config,
    model: CorefScorer,
    span_predictor: SpanPredictor,
    data_split: str = "dev",
    word_level_conll: bool = False
) -> Tuple[float, Tuple[float, float, float]]:

    w_checker = ClusterChecker()
    s_checker = ClusterChecker()
    docs = get_docs(config.__dict__[f"{data_split}_data"],
                    config.bert_model)
    data_provider = doc2inputs()
    span_provider = doc2spaninfo()
    data_provider.initialize()
    span_provider.initialize()
    running_loss = 0.0
    s_correct = 0
    s_total = 0

    with conll.open_(config, model.attrs['epochs_trained'], data_split) \
            as (gold_f, pred_f):
        pbar = tqdm.tqdm(docs, unit="docs", ncols=0)
        for doc in pbar:
            (word_features, cluster_ids), _ = data_provider(doc, False)
            # Get data for SpanPredictor
            (heads, starts, ends), _ = span_provider(doc, False)
            # Run CorefScorer
            coref_scores, top_indices = model.predict((doc, word_features[0], cluster_ids))
            # Compute coreference loss
            c_loss, c_grads = coref_loss(
                span_provider,
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
                        doc,
                        word_features[0],
                        heads
                    )
                )
                span_clusters = predict_span_clusters(
                    span_predictor,
                    doc,
                    word_features[0],
                    word_clusters
                )
                pred_starts = span_scores[:, :, 0].argmax(axis=1)
                pred_ends = span_scores[:, :, 1].argmax(axis=1)
                s_correct += ((starts == pred_starts) * (ends == pred_ends)).sum()
                s_total += len(pred_starts)

            if word_level_conll:
                conll.write_conll(doc,
                                  [[(i, i + 1) for i in cluster]
                                   for cluster in doc["word_clusters"]],
                                  gold_f)
                conll.write_conll(doc,
                                  [[(i, i + 1) for i in cluster]
                                   for cluster in word_clusters],
                                  pred_f)
            else:
                conll.write_conll(doc, doc["span_clusters"], gold_f)
                conll.write_conll(doc, span_clusters, pred_f)

            w_checker.add_predictions(doc["word_clusters"], word_clusters)
            w_lea = w_checker.total_lea

            s_checker.add_predictions(doc["span_clusters"], span_clusters)
            s_lea = s_checker.total_lea


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
        print()
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
