import numpy as np
import itertools
import random

# ------------------------------
# Common Helper Functions
# ------------------------------


# Activation functions
def binary_step(x):
    return 1 if x >= 0 else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# Generate all binary input combinations
def generate_inputs(n):
    return np.array(list(itertools.product([0, 1], repeat=n)))


# Get user-defined targets
def get_targets(x, output_neurons=1):
    y = []
    print("\nEnter output values (0/1) for each input:")
    for xi in x:
        if output_neurons == 1:
            try:
                out = int(input(f"{list(xi)} → "))
                y.append([out if out in (0, 1) else 0])
            except:
                y.append([0])
        else:
            row = []
            for o in range(output_neurons):
                try:
                    out = int(input(f"{list(xi)} → Output {o+1}: "))
                    row.append(out if out in (0, 1) else 0)
                except:
                    row.append(0)
            y.append(row)
    return np.array(y)


# ------------------------------
# 1. Random Value MLP - 2 Hidden Layers - 1 Output (Binary Step)
# ------------------------------
def random_mlp_two_hidden_one_output():
    print("\n--- Random MLP: Two Hidden Layers, One Output ---\n")
    n = int(input("Enter number of binary inputs: "))
    x_data = generate_inputs(n)
    y_data = get_targets(x_data)

    tries = 0
    solved = False

    while not solved:
        tries += 1

        w1 = np.random.uniform(-2, 2, (n, 2))
        w2 = np.random.uniform(-2, 2, (2, 2))
        w3 = np.random.uniform(-2, 2, (2, 1))
        b1, b2, b3 = random.randint(-3, 3), random.randint(-3, 3), random.randint(-3, 3)

        # Forward pass
        h1 = np.vectorize(binary_step)(np.dot(x_data, w1) + b1)
        h2 = np.vectorize(binary_step)(np.dot(h1, w2) + b2)
        out = np.vectorize(binary_step)(np.dot(h2, w3) + b3)

        if np.array_equal(out, y_data):
            solved = True
            print(f"\n✅ Match found after {tries} tries:")
            print("Bias 1:", b1)
            print("Bias 2:", b2)
            print("Bias 3:", b3)
            print("W1:\n", w1)
            print("W2:\n", w2)
            print("W3:\n", w3)
            for i, y, o in zip(x_data, y_data, out):
                print(f"{list(i)} → Expected: {y[0]}, Output: {int(o[0])}")
        if tries % 100000 == 0:
            print(f"Attempt {tries}... still searching.")


# ------------------------------
# 2. Random Value MLP - 1 Hidden Layer - 2 Outputs (Binary Step)
# ------------------------------
def random_mlp_one_hidden_two_outputs():
    print("\n--- Random MLP: One Hidden Layer, Two Outputs ---\n")
    n = 4
    x_data = generate_inputs(n)
    y_data = get_targets(x_data, output_neurons=2)

    tries = 0
    solved = False

    while not solved:
        tries += 1

        w1 = np.random.uniform(-2, 2, (n, 3))
        w2 = np.random.uniform(-2, 2, (3, 2))
        b1, b2 = random.randint(-3, 3), random.randint(-3, 3)

        # Forward pass
        h1 = np.vectorize(binary_step)(np.dot(x_data, w1) + b1)
        out = np.vectorize(binary_step)(np.dot(h1, w2) + b2)

        if np.array_equal(out, y_data):
            solved = True
            print(f"\n✅ Match found after {tries} tries:")
            print("Bias 1:", b1)
            print("Bias 2:", b2)
            print("W1:\n", w1)
            print("W2:\n", w2)
            for i, y, o in zip(x_data, y_data, out):
                print(f"{list(i)} → Expected: {list(y)}, Output: {list(o)}")
        if tries % 100000 == 0:
            print(f"Attempt {tries}... still searching.")


# ------------------------------
# 3, 4, 5. MLP Class with Backpropagation
# ------------------------------
class SimpleMLP:
    def __init__(self, input_size, h1_size, h2_size, activation, lr=0.1):
        self.lr = lr
        self.activation = activation

        self.w1 = np.random.randn(input_size, h1_size)
        self.b1 = np.zeros((1, h1_size))

        self.w2 = np.random.randn(h1_size, h2_size)
        self.b2 = np.zeros((1, h2_size))

        self.w3 = np.random.randn(h2_size, 1)
        self.b3 = np.zeros((1, 1))

        # Set activation functions
        if activation == "sigmoid":
            self.act = sigmoid
            self.act_derivative = sigmoid_derivative
        elif activation == "relu":
            self.act = relu
            self.act_derivative = relu_derivative
        elif activation == "tanh":
            self.act = tanh
            self.act_derivative = tanh_derivative

    def forward(self, x):
        self.z1 = self.act(np.dot(x, self.w1) + self.b1)
        self.z2 = self.act(np.dot(self.z1, self.w2) + self.b2)
        self.output = self.act(np.dot(self.z2, self.w3) + self.b3)
        return self.output

    def backward(self, x, y):
        error = y - self.output
        d_output = error * self.act_derivative(self.output)

        error2 = d_output.dot(self.w3.T)
        d_z2 = error2 * self.act_derivative(self.z2)

        error1 = d_z2.dot(self.w2.T)
        d_z1 = error1 * self.act_derivative(self.z1)

        # Update weights and biases
        self.w3 += self.z2.T.dot(d_output) * self.lr
        self.b3 += np.sum(d_output, axis=0, keepdims=True) * self.lr

        self.w2 += self.z1.T.dot(d_z2) * self.lr
        self.b2 += np.sum(d_z2, axis=0, keepdims=True) * self.lr

        self.w1 += x.T.dot(d_z1) * self.lr
        self.b1 += np.sum(d_z1, axis=0, keepdims=True) * self.lr

    def train(self, x, y, epochs=10000):
        for i in range(epochs):
            self.forward(x)
            self.backward(x, y)
            if i % 1000 == 0:
                loss = np.mean((y - self.output) ** 2)
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, x):
        return (self.forward(x) > 0.5).astype(int)


def backprop_mlp(activation_type):
    print(
        f"\n--- Backpropagation MLP: Activation = {activation_type.capitalize()} ---\n"
    )
    n = int(input("Enter number of binary inputs: "))
    lr = float(input("Enter learning rate: "))
    hidden1 = int(input("Enter neurons in Hidden Layer 1: "))
    hidden2 = int(input("Enter neurons in Hidden Layer 2: "))

    x_data = generate_inputs(n)
    y_data = get_targets(x_data)

    model = SimpleMLP(n, hidden1, hidden2, activation=activation_type, lr=lr)
    model.train(x_data, y_data)

    print("\nResults after training:")
    preds = model.predict(x_data)
    for i, y, o in zip(x_data, y_data, preds):
        print(f"{list(i)} → Expected: {y[0]}, Predicted: {o[0]}")


# ------------------------------
# Menu to Run
# ------------------------------
if __name__ == "__main__":
    while True:
        print("\nSelect Assignment:")
        print("1. Random MLP (2 Hidden Layers, 1 Output)")
        print("2. Random MLP (1 Hidden Layer, 2 Outputs)")
        print("3. Backpropagation MLP (Sigmoid)")
        print("4. Backpropagation MLP (ReLU)")
        print("5. Backpropagation MLP (Tanh)")
        print("6. Exit")

        choice = input("\nEnter choice: ")

        if choice == "1":
            random_mlp_two_hidden_one_output()
        elif choice == "2":
            random_mlp_one_hidden_two_outputs()
        elif choice == "3":
            backprop_mlp("sigmoid")
        elif choice == "4":
            backprop_mlp("relu")
        elif choice == "5":
            backprop_mlp("tanh")
        elif choice == "6":
            break
        else:
            print("Invalid choice. Try again!")


# --------------------------------------------------------------
# 🎯 Multi-Layer Perceptron (MLP) - Quick Reference Cheat Sheet
# --------------------------------------------------------------

# 📚 What is MLP?
# - A type of neural network with one or more hidden layers.
# - Each layer has neurons, each neuron has weights and biases.
# - Neurons apply activation functions to produce output.

# 🔥 Basic Steps:
# 1. Forward Propagation: Inputs → Hidden Layers → Output Layer
# 2. Activation Functions applied at each neuron.
# 3. Compare Output with Target → Calculate Error.
# 4. Backward Propagation: Update Weights and Biases (only in learning models).

# 🧠 Important Concepts:
# - Inputs: Binary inputs (0 or 1)
# - Weights: Strength of connection between neurons.
# - Bias: Adjustment term to shift activation.
# - Activation Functions:
#     - Binary Step: 0 or 1 output.
#     - Sigmoid: Smooth output between 0 and 1.
#     - ReLU: Output is input if >0, else 0.
#     - Tanh: Output between -1 and +1.
# - Epoch: One full pass over the entire dataset.

# 🚀 Differences in Assignments:

# 1. Random MLP (2 Hidden Layers, 1 Output):
# - Randomly assign weights and biases.
# - No learning (no backpropagation).
# - Keep trying random values until the desired output is achieved.

# 2. Random MLP (1 Hidden Layer, 2 Outputs):
# - Similar to (1), but with two output neurons.
# - Still random trial, no learning.

# 3. Backpropagation MLP with Sigmoid Activation:
# sigma(x) = 1 / (1 + e ^ (-x))
# - Proper learning using backpropagation.
# - Activation function is Sigmoid.
# - Weights and biases are updated based on error gradients.

# 4. Backpropagation MLP with ReLU Activation:
# reLu(x) = max(0, x)
# - Learning using backpropagation.
# - Activation function is ReLU.
# - Faster convergence in many cases compared to Sigmoid.

# 5. Backpropagation MLP with Tanh Activation:
# tanh(x) = ((e^x) - (e^-x)) / ((e^x) + (e^-x))
# - Learning using backpropagation.
# - Activation function is Tanh.
# - Output centered between -1 and +1 for better balance.

# ✨ Summary:
# - Random Models = Random Trial and Error (no real learning).
# - Backpropagation Models = Systematic Learning (reduces error over time).
# - Activation Functions decide the neuron behavior and impact learning speed.
# - Weights and Biases control how the MLP "learns" from inputs.

# 🌟 Bonus Tip:
# - Sigmoid: Good for simple binary outputs.
# - ReLU: Good for deeper and faster models.
# - Tanh: Good when needing zero-centered outputs.

# --------------------------------------------------------------
# END OF CHEAT SHEET
# --------------------------------------------------------------
