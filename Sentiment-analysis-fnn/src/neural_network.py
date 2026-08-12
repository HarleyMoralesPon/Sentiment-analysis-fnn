import numpy as np

# ==========================================
# Class: FeedForwardNeuralNetwork for binary classification
# ==========================================
class FeedForwardNeuralNetwork:

# ==========================================
# initialize the neural network with the given parameters
# ==========================================
    def __init__(
        self,
        input_size,
        hidden_size=32,
        output_size=1,
        learning_rate=0.01,
        random_state=42
    ): 


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        np.random.seed(random_state)

        
        self.initialize_parameters()

    def initialize_parameters(self):
        
        # Hidden layer
        self.W1 = np.random.randn(
            self.input_size,
            self.hidden_size
        ) * 0.01

        self.b1 = np.zeros((1, self.hidden_size))

        # Output layer
        self.W2 = np.random.randn(
            self.hidden_size,
            self.output_size
        ) * 0.01

        self.b2 = np.zeros((1, self.output_size))

# ==========================================
# Activation function
# ==========================================

    def sigmoid(self, Z):

        return 1 / (1 + np.exp(-Z))


# ==========================================
# Forward propagation 
# ==========================================

    def forward(self, X):

        self.Z1 = np.dot(X, self.W1) + self.b1

        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2

        self.A2 = self.sigmoid(self.Z2)

        return self.A2


####################################################################
# LOSS FUNCTION
####################################################################

    def compute_loss(self, y_true, y_pred):

        epsilon = 1e-15

        y_pred = np.clip(
            y_pred,
            epsilon,
            1 - epsilon
        )

        loss = -np.mean(

            y_true * np.log(y_pred)

            +

            (1 - y_true) * np.log(1 - y_pred)

        )

        return loss


####################################################################
# BACKPROPAGATION
####################################################################

    def backward(self, X, y_true):

        m = X.shape[0]

        # Output layer
        self.dZ2 = self.A2 - y_true

        self.dW2 = (1 / m) * np.dot(
            self.A1.T,
            self.dZ2
        )

        self.db2 = (1 / m) * np.sum(
            self.dZ2,
            axis=0,
            keepdims=True
        )

        # Hidden layer
        self.dA1 = np.dot(
            self.dZ2,
            self.W2.T
        )

        self.dZ1 = self.dA1 * self.A1 * (1 - self.A1)

        self.dW1 = (1 / m) * np.dot(
            X.T,
            self.dZ1
        )

        self.db1 = (1 / m) * np.sum(
            self.dZ1,
            axis=0,
            keepdims=True
        )

####################################################################
# GRADIENT DESCENT
####################################################################

    def update_parameters(self):

        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1

        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2

####################################################################
# TRAINING
####################################################################

    def train(self, X, y, iterations=1000):

        self.loss_history = []

        for i in range(iterations):

            # Forward
            predictions = self.forward(X)

            # Loss
            loss = self.compute_loss(
                y,
                predictions
            )

            self.loss_history.append(loss)

            # Backward
            self.backward(
                X,
                y
            )

            # Gradient Descent
            self.update_parameters()

            if i % 100 == 0:

                print(
                    f"Iteration {i:4d} | Loss = {loss:.6f}"
                )
####################################################################
# PREDICT PROBABILITIES
####################################################################

    def predict_proba(self, X):

        return self.forward(X)

####################################################################
# PREDICT CLASSES
####################################################################

    def predict(self, X):

        probabilities = self.predict_proba(X)

        predictions = (probabilities >= 0.5).astype(int)

        return predictions