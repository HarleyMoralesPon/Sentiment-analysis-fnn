from collections import Counter


# ==========================================
# 1. function 1: build vocabulary
# ==========================================

def build_vocabulary(texts, min_frequency=1):
    """
    Build a vocabulary from a collection of texts.

    Parameters
    ----------
    texts : iterable
        Collection of cleaned tweets.

    min_frequency : int
        Minimum number of occurrences for a word
        to be included.

    Returns
    -------
    vocabulary : list
    """

    counter = Counter()

    for text in texts:

        words = text.split()

        counter.update(words)

    vocabulary = [

        word

        for word, count in counter.items()

        if count >= min_frequency

    ]

    vocabulary.sort()

    return vocabulary



# ==========================================
# function 2:  word to index
# ==========================================


def create_word_to_index(vocabulary):
    """
    Create a mapping from word to integer index.
    """

    return {

        word: idx

        for idx, word in enumerate(vocabulary)

    }





# ==========================================
# function 3: Index to Word
# ==========================================

def create_index_to_word(vocabulary):
    """
    Create a mapping from integer index to word.
    """

    return {

        idx: word   

        for idx, word in enumerate(vocabulary)

    }