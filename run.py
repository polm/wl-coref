""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""
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

from coref import CorefModel, conll
from coref.cluster_checker import ClusterChecker
from coref.const import Doc
from coref.loss import CorefLoss
from thinc.api import require_gpu

from coref.spacy_util import save_state, load_state, get_docs


def train(
    model: CorefModel,
    optimizer,
):
    """
    Trains all the trainable blocks in the model using the config provided.
    """
    docs = list(get_docs(model.config.train_data,
                         model.config.bert_model))
    n_docs = len(docs)
    docs_ids = list(range(len(docs)))
    avg_spans = sum(len(doc["head2span"]) for doc in docs) / len(docs)
    coref_criterion = CorefLoss(model.config.bce_loss_weight)
    span_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    best_val_score = 0
    # Set up optimizers

    # XXX changed
    for epoch in range(model.epochs_trained, model.config.train_epochs):
        # XXX changed
        model.training = True
        running_c_loss = 0.0
        running_s_loss = 0.0
        random.shuffle(docs_ids)
        pbar = tqdm.tqdm(docs_ids, unit="docs", ncols=0)
        for doc_id in pbar:
            doc = docs[doc_id]
            optimizer.zero_grad()

            res = model.run(doc)

            c_loss = coref_criterion(res.coref_scores, res.coref_y)
            if res.span_y:
                s_loss = (span_criterion(res.span_scores[:, :, 0], res.span_y[0])
                          + span_criterion(res.span_scores[:, :, 1], res.span_y[1])) / avg_spans / 2
            else:
                s_loss = torch.zeros_like(c_loss)

            del res

            (c_loss + s_loss).backward()
            running_c_loss += c_loss.item()
            running_s_loss += s_loss.item()

            del c_loss, s_loss

            optimizer.step()

            pbar.set_description(
                f"Epoch {epoch + 1}:"
                f" {doc['document_id']:26}"
                f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
            )

        model.epochs_trained += 1
        val_score = evaluate(model)
        if val_score > best_val_score:
            best_val_score = val_score
            print("New best {}".format(best_val_score))
            save_state(model, optimizer)


@torch.no_grad()
def evaluate(model,
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
    model.training = False
    w_checker = ClusterChecker()
    s_checker = ClusterChecker()
    coref_criterion = CorefLoss(model.config.bce_loss_weight)
    docs = get_docs(model.config.__dict__[f"{data_split}_data"],
                    model.config.bert_model)
    running_loss = 0.0
    s_correct = 0
    s_total = 0

    with conll.open_(model.config, model.epochs_trained, data_split) \
            as (gold_f, pred_f):
        pbar = tqdm.tqdm(docs, unit="docs", ncols=0)
        for doc in pbar:
            res = model.run(doc)

            running_loss += coref_criterion(res.coref_scores, res.coref_y).item()

            if res.span_y:
                pred_starts = res.span_scores[:, :, 0].argmax(dim=1)
                pred_ends = res.span_scores[:, :, 1].argmax(dim=1)
                s_correct += ((res.span_y[0] == pred_starts) * (res.span_y[1] == pred_ends)).sum().item()
                s_total += len(pred_starts)

            if word_level_conll:
                conll.write_conll(doc,
                                  [[(i, i + 1) for i in cluster]
                                   for cluster in doc["word_clusters"]],
                                  gold_f)
                conll.write_conll(doc,
                                  [[(i, i + 1) for i in cluster]
                                   for cluster in res.word_clusters],
                                  pred_f)
            else:
                conll.write_conll(doc, doc["span_clusters"], gold_f)
                conll.write_conll(doc, res.span_clusters, pred_f)

            w_checker.add_predictions(doc["word_clusters"], res.word_clusters)
            w_lea = w_checker.total_lea

            s_checker.add_predictions(doc["span_clusters"], res.span_clusters)
            s_lea = s_checker.total_lea

            del res

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
    model = CorefModel(args.config_file, args.experiment)

    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size

    if args.mode == "train":
        modules = sorted((key, value) for key,
                         value in model.trainable.items()
                         if key != "bert")
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)

        optimizer = torch.optim.Adam(
            params, lr=model.config.learning_rate)
        
        if args.weights is not None or args.warm_start:
            model, optimizer = load_state(
                model,
                optimzer=optimizer, 
                schedupath=args.weights,
                map_location="cpu",
                noexception=args.warm_start
            )
        with output_running_time():
            train(model, optimizer)
    else:
        load_state(
            model,
            path=args.weights,
            map_location="cpu",
            ignore={"optimizer"}
        )
        evaluate(model,
                 data_split=args.data_split,
                 word_level_conll=args.word_level)
