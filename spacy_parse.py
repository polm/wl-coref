import os
import spacy

from typing import Dict, List, TextIO
from convert_to_jsonlines import get_conll_filenames


def doc2conll(doc: spacy.tokens.Doc) -> str:
    for i, token in enumerate(doc):
        line = '{}\t{}\t_\t{}\t{}\t_\t{}\t{}\t_\t_'.format(
        i, token.text, token.tag_, token.tag_, token.head.i, token.dep_
        )
        print(line)
        break


def parse_sents(fileobj: TextIO) -> spacy.tokens.Doc:
    current_sent = []
    docs = []
    lines = fileobj.readlines()
    for line in lines:
        columns = line.split()
        if len(columns) == 0:
            text = " ".join(current_sent)
            doc2conll((nlp(text)))
            current_sent = []
        elif '#' in columns[0]:
            continue
        else:
            word_number, word, pos, parse_bit = columns[2:6]
            current_sent.append(word)


def parse_files(dest_dir: str,
                filenames: Dict[str, List[str]]) -> None:
    """
    Creates files names like filenames in dest_dir, writing to each file
    constituency trees line by line.
    """
    for filelist in filenames.values():
        for filename in filelist:
            with open(filename, mode="r", encoding="utf8") as f_to_read:
                temp_path = os.path.join(dest_dir, filename)
                assert not os.path.isfile(temp_path)
                temp_dir = os.path.split(temp_path)[0]
                os.makedirs(temp_dir, exist_ok=True)
                with open(temp_path, mode="w", encoding="utf8") as f_to_write:
                    for doc in parse_sents(f_to_read):
                        doc2conll(doc)


nlp = spacy.load("en_core_web_trf")
filenames = get_conll_filenames('./data/conll-2012/v4/data/',
                                'english')
print(filenames['train'][0])
with open(filenames['train'][0]) as f:
    parse_sents(f)
