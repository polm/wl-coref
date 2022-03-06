"""
This module just takes the generated Doc objects
from the wl-coref library and converts them
to spacy.tokens.Doc
"""
import os
import logging
import jsonlines
import tqdm
import spacy

from collections import defaultdict
from typing import List, Dict, Tuple
from spacy.tokens import Doc, Token, Span, SpanGroup, DocBin

DATA_DIR = "."
FILENAME = "data/english_{}_head.jsonlines"
LOGGING_LEVEL = logging.WARNING  # DEBUG to output all duplicate spans
SPLITS = ("development", "test", "train")
nlp = spacy.blank("en")


def get_gold_word_clusters(doc):
    """Return world-level clusters as a list of lists of token indices."""
    clusters = []
    for key, sg in doc.spans.items():
        if not key.startswith("coref_word_clusters_"):
            continue

        # it's a group of spans, but each span is just one token
        cluster = sorted([span[0].i for span in sg])
        clusters.append(cluster)
    return clusters

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

def load_spacy_data(path):
    return list(
        DocBin(
            store_user_data=True
        ).from_disk(
            path
        ).get_docs(
            nlp.vocab
        )
    )
if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    path = os.path.join(DATA_DIR, FILENAME)
    for split in SPLITS:
        with jsonlines.open(path.format(split, ""), mode='r') as inf:
            # inf = pickle.load(f)
            doc_bin = DocBin(store_user_data=True)
            for doc in tqdm.tqdm(inf):
                sent_starts = _sentids_to_sentstarts(doc['sent_id'])
                # heads are None for root, but should be token's own index
                fixed_heads = [(hh or ii) for ii, hh in enumerate(doc['head'])]
                spacy_doc = Doc(
                    vocab=nlp.vocab,
                    words=doc['cased_words'],
                    sent_starts=sent_starts,
                    heads=fixed_heads,
                    deps=doc['deprel'],
                )
                #spacy_doc._.document_id = doc['document_id']
                #spacy_doc._.part_id = doc['part_id']
                #spacy_doc._.coref_head2span = doc['head2span']
                #assert len(spacy_doc) == len(doc['sent_id'])

                
                old_sent = -1
                for i, sid in enumerate(doc['sent_id']):
                    #spacy_doc[i].is_sent_start = (sid != old_sent)
                    old_sent = sid
                for i, cluster in enumerate(doc['word_clusters']):
                    sg = []
                    for position in cluster:
                        sg.append(spacy_doc[position:position+1])
                        #token = spacy_doc[position]
                        #token._.coref_cluster_i = i
                    if len(sg) == 0:
                        print("WARN: Empty word cluster!")
                    spacy_doc.spans[f"coref_word_clusters_{i}"] = sg

                gold_word_clusters = get_gold_word_clusters(spacy_doc)
                mention_spans = []
                for i, cluster in enumerate(doc['span_clusters']):
                    sg = []
                    for start, end in cluster:
                        mention_span = spacy_doc[start:end]
                        sg.append(mention_span)

                    spacy_doc.spans[f"coref_clusters_{i}"] = sg
                doc_bin.add(spacy_doc)

            if split == "development":
                split = 'dev'
            doc_bin.to_disk("{}.spacy".format(split))
