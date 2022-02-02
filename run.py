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
import transformers     # type: ignore

from coref import CorefScorer, SpanPredictor, conll
from coref.cluster_checker import ClusterChecker
from coref.const import Doc
from coref.loss import CorefLoss
from thinc.api import require_gpu
from thinc.api import PyTorchWrapper
from thinc.api import Adam as Tadam
from coref.spacy_util import save_state, load_state, get_docs, _clusterize, _load_config
from coref.thinc_funcs import doc2inputs, doc2spaninfo, convert_coref_scorer_inputs

from coref.thinc_loss import coref_loss


def train(
    config,
    model: CorefScorer,
    span_predictor: SpanPredictor,
    optimizer,
):
    """
    Trains all the trainable blocks in the model using the config provided.
    """
    docs = list(get_docs(config.train_data,
                         config.bert_model))
    data_provider = doc2inputs()
    span_provider = doc2spaninfo()
    thinc_optimizer = Tadam(config.learning_rate)
    data_provider.initialize()
    model.initialize()
    n_docs = len(docs)
    docs_ids = list(range(len(docs)))
    avg_spans = sum(len(doc["head2span"]) for doc in docs) / len(docs)
    coref_criterion = CorefLoss(config.bce_loss_weight)
    span_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    best_val_score = 0

    for epoch in range(model.attrs['epochs_trained'], config.train_epochs):
        # model.train()
        span_predictor.train()
        running_c_loss = 0.0
        running_s_loss = 0.0
        random.shuffle(docs_ids)
        pbar = tqdm.tqdm(docs_ids, unit="docs", ncols=0)
        for doc_id in pbar:
            doc = docs[doc_id]
            optimizer.zero_grad()
            # Get data for CorefScorer
            (word_features, cluster_ids), _ = data_provider(doc, False)
            # Get data for SpanPredictor
            (heads, starts, ends), _ = span_provider(doc, False)
            starts = torch.tensor(starts).to(span_predictor.device)
            ends = torch.tensor(ends).to(span_predictor.device)
            span_target = (starts, ends)
            # Run CorefScorer
            (coref_scores, top_indices), backprop = model.begin_update((doc, word_features[0], cluster_ids))
            # Compute coreference loss
            # c_loss = coref_criterion(
            #    cluster_ids,
            #    coref_scores,
            #    top_indices
            # )
            c_loss, c_grads = coref_loss(
                span_provider,
                cluster_ids,
                coref_scores,
                top_indices,
                False
            )
            backprop(c_grads)
            model.finish_update(optimizer)
            # Run SpanPredictor
            if starts.nelement() and ends.nelement():
                span_scores = span_predictor(
                    doc,
                    word_features[0],
                    heads
                )
                s_loss = (span_criterion(span_scores[:, :, 0], span_target[0])
                          + span_criterion(span_scores[:, :, 1], span_target[1])) / avg_spans / 2
                del span_scores
            else:
                s_loss = torch.zeros_like(c_loss)

            (c_loss + s_loss).backward()
            running_c_loss += c_loss.item()
            running_s_loss += s_loss.item()

            del coref_scores
            del top_indices
            del c_loss, s_loss

            optimizer.step()

            pbar.set_description(
                f"Epoch {epoch + 1}:"
                f" {doc['document_id']:26}"
                f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
            )

        model.attrs['epochs_trained'] += 1
        val_score = evaluate(model, span_predictor)
        if val_score > best_val_score:
            best_val_score = val_score
            print("New best {}".format(best_val_score))
            save_state(model, span_predictor, optimizer)


@torch.no_grad()
def evaluate(
    config,
    model: CorefScorer,
    span_predictor: SpanPredictor,
    data_split: str = "dev",
    word_level_conll: bool = False
) -> Tuple[float, Tuple[float, float, float]]:
    """ Evaluates the modes on the data split provided.

    Args:
        data_split (str): one of 'dev'/'test'/'train'
        word_level_conll (bool): if True, outputs conll files on word-level

    Returns:
        mean loss
        span-level LEA: f1, precision, recal
    """
    model.eval()
    span_predictor.eval()
    w_checker = ClusterChecker()
    s_checker = ClusterChecker()
    coref_criterion = CorefLoss(config.bce_loss_weight)
    docs = get_docs(config.__dict__[f"{data_split}_data"],
                    config.bert_model)
    data_provider = doc2inputs()
    span_provider = doc2spaninfo()
    data_provider.initialize()
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
            starts = torch.tensor(starts).to(span_predictor.device)
            ends = torch.tensor(ends).to(span_predictor.device)
            span_target = (starts, ends)
            # Run CorefScorer
            coref_scores, top_indices = model(
                doc,
                word_features[0],
                cluster_ids)
            # Compute coreference loss
            c_loss = coref_criterion(
                cluster_ids,
                coref_scores,
                top_indices
            )
            word_clusters = _clusterize(
                 coref_scores,
                 top_indices
             )
            running_loss += c_loss.item()
            if starts.nelement() and ends.nelement():
                span_scores = span_predictor(
                    doc,
                    word_features[0],
                    heads
                )
                span_clusters = span_predictor.predict(
                    doc,
                    word_features[0],
                    word_clusters
                )
                pred_starts = span_scores[:, :, 0].argmax(dim=1)
                pred_ends = span_scores[:, :, 1].argmax(dim=1)
                s_correct += ((span_target[0] == pred_starts) * (span_target[1] == pred_ends)).sum().item()
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
    argparser.add_argument("--warm-start", action="store_true",
                           help="If set, the training will resume from the"
                                " last checkpoint saved if any. Ignored in"
                                " evaluation modes."
                                " Incompatible with '--weights'.")
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--word-level", action="store_true",
                           help="If set, output word-level conll-formatted"
                                " files in evaluation modes. Ignored in"
                                " 'train' mode.")
    args = argparser.parse_args()

    if args.warm_start and args.weights is not None:
        print("The following options are incompatible:"
              " '--warm_start' and '--weights'", file=sys.stderr)
        sys.exit(1)

    require_gpu()
    seed(2020)
    config = _load_config(args.config_file, args.experiment)
    model = PyTorchWrapper(
        CorefScorer(
            config.device,
            config.embedding_size,
            config.hidden_size,
            config.n_hidden_layers,
            config.dropout_rate,
            config.rough_k,
            config.a_scoring_batch_size
        ), convert_inputs=convert_coref_scorer_inputs
    )
    span_predictor = SpanPredictor(
        1024,
        config.sp_embedding_size).to(config.device)
    if args.batch_size:
        config.a_scoring_batch_size = args.batch_size

    if args.mode == "train":
        parameters = span_predictor.parameters()
        optimizer = torch.optim.Adam(
            parameters, lr=config.learning_rate)

        if args.weights is not None or args.warm_start:
            model, optimizer = load_state(
                model,
                optimzer=optimizer,
                schedupath=args.weights,
                map_location="cpu",
                noexception=args.warm_start
            )
        else:
            model.attrs['epochs_trained'] = 0
        with output_running_time():
            train(config, model, span_predictor, optimizer)
    else:
        load_state(
            model,
            span_predictor,
            path=args.weights,
            map_location="cpu",
        )
        evaluate(config,
                 model,
                 span_predictor,
                 data_split=args.data_split,
                 word_level_conll=args.word_level)
