"""
These are just hacks at the moment, but this 
is the place to implement spaCy functions for
data loading.
"""

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
