""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""

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

from coref import CorefModel
from coref.const import Doc
from thinc.api import require_gpu

from coref.spacy_util import save_state, load_state


def train(
    model: CorefModel,
    optimizer,
):
    """
    Trains all the trainable blocks in the model using the config provided.
    """
    docs = list(model._get_docs(model.config.train_data))
    docs = docs[:5]
    n_docs = len(docs)
    docs_ids = list(range(len(docs)))
    avg_spans = sum(len(doc["head2span"]) for doc in docs) / len(docs)
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

            c_loss = model._coref_criterion(res.coref_scores, res.coref_y)
            if res.span_y:
                s_loss = (model._span_criterion(res.span_scores[:, :, 0], res.span_y[0])
                          + model._span_criterion(res.span_scores[:, :, 1], res.span_y[1])) / avg_spans / 2
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
        val_score = model.evaluate()
        #if val_score > best_val_score:
        best_val_score = val_score
        print("New best {}".format(best_val_score))
        save_state(model, optimizer)

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
        model.evaluate(data_split=args.data_split,
                       word_level_conll=args.word_level)
