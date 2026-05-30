# -*- coding: utf-8 -*-
import copy
import pickle

SPECIAL_TOKENS = {
    '<PAD>': 0,     # Padding token - to equalize size of expressions
    '<EOS>': 1,     # End of sentence token - to mark completion
    '<UNK>': 2,     # Unknown token - Replacement when out of vocab characters occur
    '<GO>' : 3      # Start of sentence token - to mark initiation
}

"""
To create lookup tables for vocabulary
"""
def lookup_table(vocabulary):
    """
    vocabulary: String of all characters of a particular language
    """
    vocabulary = set(list(vocabulary))
    v2i = copy.copy(SPECIAL_TOKENS)
    for i, vocab in enumerate(vocabulary, len(SPECIAL_TOKENS)):
        v2i[vocab] = i

    i2v = {i:v for v, i in v2i.items()}
    return v2i, i2v


def text2id(words, v2i, target=False):
     """
     words: List of words to be converted to indices
     v2i: Lookup table for the same language
     target: Tells if the list of words are target words
     """
     text_ids = []
     max_length = max([len(word) for word in words])

     for word in words:
         tokens = list(word)
         token_ids = []
         for token in tokens:
             token_ids.append(v2i[token])

        # If my words are target words, we need to add the EOS special token to identify when to stop creating a sequence
         if target:
             token_ids.append(v2i['<EOS>'])

         text_ids.append(token_ids)

     return text_ids

"""
To preprocess the text
"""
def preprocess(src_text, target_text):
    ENG = 'abcdefghijklmnopqrstuvwxyz'
    HIN = 'ँंॉॆॊॏऺऻॎःािीुूेैोौअआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहज्ञक्षश्रज़रफ़ड़ढ़ख़क़ग़ळृृ़़ऑ'

    src_v2i, src_i2v = lookup_table(HIN)
    target_v2i, target_i2v = lookup_table(ENG)

    src_text = text2id(src_text, src_v2i)
    target_text = text2id(target_text, target_v2i, target=True)

    pickle.dump(((src_text, target_text),
                (src_v2i, src_i2v),
                (target_v2i, target_i2v)
                ), open('preprocess.p', 'wb'))

    return src_text, target_text, src_v2i, src_i2v, target_v2i, target_i2v


"""
To load the pickle file
"""
def load_preprocess_pickle():
    with open('preprocess.p', mode='rb') as f:
        return pickle.load(f)
