import re
import pandas as pd 

def load_dataset(filepath):
    """
    Load the sentiment dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    df = pd.read_csv(filepath)

    return df