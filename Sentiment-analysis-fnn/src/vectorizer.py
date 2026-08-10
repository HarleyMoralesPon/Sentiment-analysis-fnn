import numpy as np

# ==========================================
# function 1: sentence to bag of words
# ==========================================

def sentence_to_bow(sentence, word_to_index):
    """
    Convert one sentence into a Bag of Words vector.
    """

    vector = np.zeros(len(word_to_index), dtype=np.float32)

    words = sentence.split()

    for word in words:

        if word in word_to_index:

            index = word_to_index[word]

            vector[index] += 1

    return vector

# ==========================================
# function 2: transform dataset - Feature matrix
# ==========================================

def transform_dataset(texts, word_to_index):
    """
    Convert a collection of texts into a feature matrix.
    """

    feature_matrix = []

    for text in texts:

        bow = sentence_to_bow(text, word_to_index)

        feature_matrix.append(bow)

    return np.array(feature_matrix)