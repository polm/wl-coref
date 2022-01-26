""" see __init__.py """

from datetime import datetime
import os
import pickle
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonlines        # type: ignore
import toml
import torch
from tqdm import tqdm   # type: ignore
import transformers     # type: ignore

from coref import conll, utils
from coref.anaphoricity_scorer import AnaphoricityScorer
from coref.cluster_checker import ClusterChecker
from coref.config import Config
from coref.const import CorefResult, Doc
from coref.loss import CorefLoss
from coref.pairwise_encoder import DistancePairwiseEncoder
from coref.rough_scorer import RoughScorer
from coref.span_predictor import SpanPredictor
from coref.tokenizer_customization import TOKENIZER_FILTERS, TOKENIZER_MAPS
from coref.utils import GraphNode
from coref.word_encoder import WordEncoder


from coref.spacy_util import _convert_to_spacy_doc, _cluster_ids, _load_config
from coref.spacy_util import _get_ground_truth, _clusterize, _tokenize_docs
from thinc.api import reduce_first
from spacy_transformers.architectures import transformer_tok2vec_v3
from spacy_transformers.span_getters import configure_strided_spans


class CorefModel:  # pylint: disable=too-many-instance-attributes
    """Combines all coref modules together to find coreferent spans.

    Attributes:
        config (coref.config.Config): the model's configuration,
            see config.toml for the details
        epochs_trained (int): number of epochs the model has been trained for
        trainable (Dict[str, torch.nn.Module]): trainable submodules with their
            names used as keys
        training (bool): used to toggle train/eval modes

    Submodules (in the order of their usage in the pipeline):
        tokenizer (transformers.AutoTokenizer)
        bert (transformers.AutoModel)
        we (WordEncoder)
        rough_scorer (RoughScorer)
        pw (PairwiseEncoder)
        a_scorer (AnaphoricityScorer)
        sp (SpanPredictor)
    """
    def __init__(self,
                 config_path: str,
                 section: str,
                 epochs_trained: int = 0):
        """
        A newly created model is set to evaluation mode.

        Args:
            config_path (str): the path to the toml file with the configuration
            section (str): the selected section of the config file
            epochs_trained (int): the number of epochs finished
                (useful for warm start)
        """
        self.config = _load_config(config_path, section)
        self.epochs_trained = epochs_trained
        self._docs: Dict[str, List[Doc]] = {}
        self._build_model()
        # self._build_optimizers()
        self._set_training(False)
        self._coref_criterion = CorefLoss(self.config.bce_loss_weight)
        self._span_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    @property
    def training(self) -> bool:
        """ Represents whether the model is in the training mode """
        return self._training

    @training.setter
    def training(self, new_value: bool):
        if self._training is new_value:
            return
        self._set_training(new_value)

    # ========================================================== Public methods

    @torch.no_grad()
    def evaluate(self,
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
        self.training = False
        w_checker = ClusterChecker()
        s_checker = ClusterChecker()
        docs = self._get_docs(self.config.__dict__[f"{data_split}_data"])
        running_loss = 0.0
        s_correct = 0
        s_total = 0

        with conll.open_(self.config, self.epochs_trained, data_split) \
                as (gold_f, pred_f):
            pbar = tqdm(docs, unit="docs", ncols=0)
            for doc in pbar:
                res = self.run(doc)

                running_loss += self._coref_criterion(res.coref_scores, res.coref_y).item()

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

    def run(self,  # pylint: disable=too-many-locals
            doc: Doc,
            ) -> CorefResult:
        """
        This is a massive method, but it made sense to me to not split it into
        several ones to let one see the data flow.

        Args:
            doc (Doc): a dictionary with the document data.

        Returns:
            CorefResult (see const.py)
        """
        # Encode words with bert
        # words           [n_words, span_emb]
        # cluster_ids     [n_words]
        # words, cluster_ids = self.we(doc, self._bertify(doc))
        bert_words = self._bert_encode(doc)
        words = self.we(bert_words)
        cluster_ids = _cluster_ids(doc, self.config.device)
        # Obtain bilinear scores and leave only top-k antecedents for each word
        # top_rough_scores  [n_words, n_ants]
        # top_indices       [n_words, n_ants]
        top_rough_scores, top_indices = self.rough_scorer(words)
        # Get pairwise features [n_words, n_ants, n_pw_features]
        pw = self.pw(top_indices, doc)
        batch_size = self.config.a_scoring_batch_size
        a_scores_lst: List[torch.Tensor] = []

        for i in range(0, len(words), batch_size):
            pw_batch = pw[i:i + batch_size]
            words_batch = words[i:i + batch_size]
            top_indices_batch = top_indices[i:i + batch_size]
            top_rough_scores_batch = top_rough_scores[i:i + batch_size]

            # a_scores_batch    [batch_size, n_ants]
            a_scores_batch = self.a_scorer(
                all_mentions=words, mentions_batch=words_batch,
                pw_batch=pw_batch, top_indices_batch=top_indices_batch,
                top_rough_scores_batch=top_rough_scores_batch
            )
            a_scores_lst.append(a_scores_batch)

        res = CorefResult()

        # coref_scores  [n_spans, n_ants]
        res.coref_scores = torch.cat(a_scores_lst, dim=0)
        res.coref_y = _get_ground_truth(
            cluster_ids, top_indices, (top_rough_scores > float("-inf")))
        res.word_clusters = _clusterize(doc, res.coref_scores,
                                             top_indices)
        res.span_scores, res.span_y = self.sp.get_training_data(doc, words)

        if not self.training:
            res.span_clusters = self.sp.predict(doc, words, res.word_clusters)

        return res

    def _bert_encode(self, doc: Doc):
        """Encode a single document."""
        spacy_doc = _convert_to_spacy_doc(doc)
        output, _ = self.bert([spacy_doc], 
                              is_train=False)
        output = torch.tensor(output[0]).to(self.config.device)
        return output
    
    def _build_model(self):
        self.pw = DistancePairwiseEncoder(self.config).to(self.config.device)
        self.bert = transformer_tok2vec_v3(name='roberta-large',
                                           get_spans=configure_strided_spans(400, 350),
                                           tokenizer_config={'add_prefix_space': True},
                                           pooling=reduce_first())
        self.bert.initialize()
        # pylint: disable=line-too-long
        bert_emb = 1024
        pair_emb = bert_emb * 3 + self.pw.shape
        self.a_scorer = AnaphoricityScorer(pair_emb, self.config).to(self.config.device)
        self.we = WordEncoder(bert_emb, self.config).to(self.config.device)
        self.rough_scorer = RoughScorer(bert_emb, self.config).to(self.config.device)
        self.sp = SpanPredictor(bert_emb, self.config.sp_embedding_size).to(self.config.device)

        self.trainable: Dict[str, torch.nn.Module] = {
            # "bert": self.bert, 
            "we": self.we,
            "rough_scorer": self.rough_scorer,
            "pw": self.pw, "a_scorer": self.a_scorer,
            "sp": self.sp
        }
    

    def _get_docs(self, path: str) -> List[Doc]:
        if path not in self._docs:
            basename = os.path.basename(path)
            model_name = self.config.bert_model.replace("/", "_")
            cache_filename = f"{model_name}_{basename}.pickle"
            if os.path.exists(cache_filename):
                with open(cache_filename, mode="rb") as cache_f:
                    self._docs[path] = pickle.load(cache_f)
            else:
                self._docs[path] = _tokenize_docs(path)
                with open(cache_filename, mode="wb") as cache_f:
                    pickle.dump(self._docs[path], cache_f)
        return self._docs[path]

    def _set_training(self, value: bool):
        self._training = value
        for module in self.trainable.values():
            module.train(self._training)
