from src.preprocessing import load_dataset
from src.preprocessing import clean_text
from src.preprocessing import preprocess_dataframe
from src.vocabulary import (build_vocabulary, 
                            create_word_to_index, 
                            create_index_to_word)
from src.vectorizer import transform_dataset    
from src.dataset import prepare_dataset 
from src.neural_network import FeedForwardNeuralNetwork
import numpy as np

# ==========================================
# 1. Load Dataset
# ==========================================
df = load_dataset ("C:\\Users\\HarleyRicardoMorales\\OneDrive - BBQ - Baumann Bildung und Qualifizierung GmbH\\Dokumente\\ML-Projects\\Sentiment-analysis-fnn\\data\\raw\\sentimentdataset.csv")

# ==========================================
# 2. Preprocess text
# ==========================================
df = preprocess_dataframe(df)
print (df.head ())


# ==========================================
# 3. Build Vocabulary
# ==========================================
vocabulary = build_vocabulary (df['Text'])
word_to_index = create_word_to_index (vocabulary)
index_to_word = create_index_to_word (vocabulary)

print(f"Vocabulary size: {len(vocabulary)}")

print(vocabulary[:20])

print(word_to_index["achievement"])

# ==========================================
# 4. Vectorize text
# ==========================================

X = transform_dataset(df["Text"], word_to_index)

print(X.shape)

print(X[:5])


# =====================================
# 5. Prepare Dataset
# =====================================

X_train, X_test, y_train, y_test = prepare_dataset(
    X,
    df
)

print("Training set")
print(X_train.shape)
print(y_train.shape)

print()

print("Testing set")
print(X_test.shape)
print(y_test.shape)

print("\nTraining distribution")

print(np.unique(y_train, return_counts=True))

print("\nTesting distribution")

print(np.unique(y_test, return_counts=True))


# -----------------------------------------
# 6. Initialize the Neural Network
# -----------------------------------------

model = FeedForwardNeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=32,
    output_size=1,
    learning_rate=0.01,
    random_state=42
)

# -----------------------------------------
# Verify the parameters
# -----------------------------------------

print("W1 shape:", model.W1.shape)
print("b1 shape:", model.b1.shape)

print("W2 shape:", model.W2.shape)
print("b2 shape:", model.b2.shape)

# -----------------------------------------
# 7. Sigmoid Activation Function and forward propagation
# -----------------------------------------

model = FeedForwardNeuralNetwork(
    input_size=X_train.shape[1]
)

predictions = model.forward(X_train)

print(predictions.shape)

print(predictions[:5])