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
