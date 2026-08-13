from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

####################################################################
# Evaluate the trained neural network
####################################################################

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the trained neural network.
    """

    train_predictions = model.predict(X_train)

    test_predictions = model.predict(X_test)

    print("=" * 50)
    print("TRAINING PERFORMANCE")
    print("=" * 50)

    print(f"Accuracy : {accuracy_score(y_train, train_predictions):.4f}")
    print(f"Precision: {precision_score(y_train, train_predictions):.4f}")
    print(f"Recall   : {recall_score(y_train, train_predictions):.4f}")
    print(f"F1 Score : {f1_score(y_train, train_predictions):.4f}")

    print()

    print("=" * 50)
    print("TEST PERFORMANCE")
    print("=" * 50)

    print(f"Accuracy : {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Precision: {precision_score(y_test, test_predictions):.4f}")
    print(f"Recall   : {recall_score(y_test, test_predictions):.4f}")
    print(f"F1 Score : {f1_score(y_test, test_predictions):.4f}")

    print()

    print("=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)

    print(confusion_matrix(y_test, test_predictions))

    print()

    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)

    print(classification_report(y_test, test_predictions))


####################################################################
# Plot the loss history
####################################################################

def plot_loss(model):

    plt.figure(figsize=(8,5))

    plt.plot(model.loss_history)

    plt.title("Training Loss")

    plt.xlabel("Iteration")

    plt.ylabel("Binary Cross Entropy")

    plt.grid(True)

    plt.show()