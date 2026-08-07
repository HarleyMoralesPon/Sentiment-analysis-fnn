import re
import pandas as pd 

'''
Function 1: load dataset
'''
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


'''
Function 2: clean text

'''
def clean_text (text):
    text = text.lower() # convert to lowercase

    text = re.sub(r"http\S+|www\S+", "", text) # remove URLS

    text = re.sub(r"@\w+", "", text) # remove mentions

    text = re.sub(r"[^\w\s]", "", text) # remove punctuation

    text = re.sub(r"\d+", "", text) # remove numbers

    text = re.sub(r"\s+", " ", text).strip() # remove extra whitespace

    return text



'''
Function 3: preprocess dataset
'''
def preprocess_dataframe(df):
    df = df.copy ()
    df["Text"] = df["Text"].apply(clean_text)
    return df

'''
- Now apply the cleaning function.
- For example, if your tweet column is called "Text":
'''