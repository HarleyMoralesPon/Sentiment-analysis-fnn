from src.preprocessing import load_dataset
from src.preprocessing import clean_text
from src.preprocessing import preprocess_dataframe
from src.vocabulary import (build_vocabulary, 
                            create_word_to_index, 
                            create_index_to_word)


df = load_dataset ("C:\\Users\\HarleyRicardoMorales\\OneDrive - BBQ - Baumann Bildung und Qualifizierung GmbH\\Dokumente\\ML-Projects\\Sentiment-analysis-fnn\\data\\raw\\sentimentdataset.csv")
df = preprocess_dataframe(df)
print (df.head ())



vocabulary = build_vocabulary (df['Text'])
word_to_index = create_word_to_index (vocabulary)
index_to_word = create_index_to_word (vocabulary)

print(f"Vocabulary size: {len(vocabulary)}")

print(vocabulary[:20])

print(word_to_index["achievement"])