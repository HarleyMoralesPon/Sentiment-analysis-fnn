import numpy as np
from sklearn.model_selection import train_test_split


def prepare_dataset(X, df):
    """
    Prepare the dataset for binary sentiment classification.

    Parameters
    ----------
    X : np.ndarray
        Bag of Words feature matrix.

    df : pandas.DataFrame
        DataFrame containing the sentiment labels.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    # ---------------------------------------
    # Normalize labels
    # ---------------------------------------

    df = df.copy()

    df["Sentiment"] = (
        df["Sentiment"]
        .str.strip()
        .str.lower()
    )
    # ---------------------------------------
    # Define binary classes
    # ---------------------------------------

    positive_labels = [
        "positive",
        "joy",
        "excitement",
        "contentment",
        "happiness",
        "love",
        "relief",
        "gratitude",
        "admiration",
        "amusement",
        "pride",
        "optimism"
    ]

    negative_labels = [
        "negative",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "boredom",
        "frustration",
        "disappointment",
        "anxiety",
        "grief",
        "loneliness",
        "despair",
        "hate"
    ]

    # ---------------------------------------
    # Keep only binary classes
    # ---------------------------------------

    mask = df["Sentiment"].isin(
        positive_labels + negative_labels
    )

    df = df[mask].copy()

    # IMPORTANT
    # X must be filtered using the same mask

    X = X[mask]

    # ---------------------------------------
    # Create target vector
    # ---------------------------------------

    df["Target"] = df["Sentiment"].apply(
        lambda x: 1 if x in positive_labels else 0
    )

    y = df["Target"].to_numpy().reshape(-1, 1)

    # ---------------------------------------
    # Train / Test Split
    # ---------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test