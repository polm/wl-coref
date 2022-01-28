from typing import Dict, List

import torch

from coref.anaphoricity_scorer import AnaphoricityScorer
from coref.const import CorefResult, Doc
from coref.pairwise_encoder import DistancePairwiseEncoder
from coref.rough_scorer import RoughScorer
from coref.span_predictor import SpanPredictor

from coref.spacy_util import _load_config
from coref.spacy_util import _get_ground_truth, _clusterize


class CorefModel(torch.nn.Module):
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
        super().__init__()
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
        self.pw = DistancePairwiseEncoder(self.config).to(self.config.device)
        bert_emb = 1024
        pair_emb = bert_emb * 3 + self.pw.shape
        self.a_scorer = AnaphoricityScorer(pair_emb, self.config).to(self.config.device)
        self.lstm = torch.nn.LSTM(input_size=bert_emb,
                                  hidden_size=bert_emb,
                                  batch_first=True,
                                  )
        self.dropout = torch.nn.Dropout(self.config.dropout_rate)
        self.rough_scorer = RoughScorer(bert_emb, self.config).to(self.config.device)
        self.sp = SpanPredictor(bert_emb, self.config.sp_embedding_size).to(self.config.device)

    def forward(
        self,
        doc: Doc,
        word_features,
        cluster_ids
) -> CorefResult:
        """
        This is a massive method, but it made sense to me to not split it into
        several ones to let one see the data flow.

        Args:
            doc (Doc): a dictionary with the document data.

        Returns:
            CorefResult (see const.py)
        """
        # words           [n_words, span_emb]
        # cluster_ids     [n_words]
        # XXX will be done with convert_inputs
        word_features = torch.tensor(word_features)
        word_features = torch.unsqueeze(word_features, dim=0)
        cluster_ids = torch.tensor(cluster_ids)
        words, _ = self.lstm(word_features)
        words = words.squeeze()
        words = self.dropout(words)
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
