from src.preprocessing import load_dataset
from src.preprocessing import clean_text
from src.preprocessing import preprocess_dataframe

df = load_dataset ("C:\\Users\\HarleyRicardoMorales\\OneDrive - BBQ - Baumann Bildung und Qualifizierung GmbH\\Dokumente\\ML-Projects\\Sentiment-analysis-fnn\\data\\raw\\sentimentdataset.csv")

df = preprocess_dataframe(df)
print (df.head ())
