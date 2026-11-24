"""Neural-network fundamentals in pure NumPy (brief section 3).

A tiny 2-layer network trained on a toy 2-D problem (two interleaved moons-like clusters)
to show: activation functions, forward propagation, loss, backpropagation, gradient descent.

Run:  python -m src.perceptron_numpy
"""
import matplotlib.pyplot as plt
import numpy as np

from src import config


# ------------------------------------------------------------ activations + derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(a):          # derivative expressed via the activation output a = sigmoid(z)
    return a * (1 - a)


def tanh(z):
    return np.tanh(z)


def d_tanh(a):
    return 1 - a ** 2


def relu(z):
    return np.maximum(0, z)


def d_relu(a):
    return (a > 0).astype(float)


ACTIVATIONS = {"sigmoid": (sigmoid, d_sigmoid), "tanh": (tanh, d_tanh), "relu": (relu, d_relu)}


# ------------------------------------------------------------ toy data
def make_toy_data(n=400, seed=0):
    """Two noisy half-circles (non-linearly separable, so a single perceptron fails)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n // 2)
    x0 = np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.12, (n // 2, 2))
    x1 = np.c_[1 - np.cos(t), 0.4 - np.sin(t)] + rng.normal(0, 0.12, (n // 2, 2))
    X = np.vstack([x0, x1])
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].reshape(-1, 1)
    return X, y


# ------------------------------------------------------------ the network
class TwoLayerNet:
    def __init__(self, n_in=2, n_hidden=8, activation="tanh", lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, 0.5, (n_hidden, 1))
        self.b2 = np.zeros((1, 1))
        self.act, self.d_act = ACTIVATIONS[activation]
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.act(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)           # output probability
        return self.a2

    @staticmethod
    def loss(y, p):
        """Binary cross-entropy."""
        eps = 1e-9
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    def backward(self, X, y):
        m = X.shape[0]
        dz2 = (self.a2 - y) / m                       # d(loss)/d(z2) for sigmoid + BCE
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ self.W2.T) * self.d_act(self.a1)  # chain rule through the hidden layer
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        # gradient descent step
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=1500):
        losses = []
        for _ in range(epochs):
            p = self.forward(X)
            losses.append(self.loss(y, p))
            self.backward(X, y)
        return losses

    def accuracy(self, X, y):
        return float(((self.forward(X) > 0.5) == y).mean())


def main():
    X, y = make_toy_data()
    plt.figure(figsize=(6, 4))
    results = {}
    for name in ACTIVATIONS:
        net = TwoLayerNet(activation=name, lr=0.1)
        losses = net.fit(X, y)
        results[name] = net.accuracy(X, y)
        plt.plot(losses, label=f"{name} (acc={results[name]:.2f})")
        print(f"{name:8s} final loss {losses[-1]:.4f}  accuracy {results[name]:.3f}")
    plt.xlabel("epoch")
    plt.ylabel("binary cross-entropy")
    plt.title("NumPy network: activation function comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR / "numpy_activation_comparison.png", dpi=150)
    print("Saved figure to outputs/figures/numpy_activation_comparison.png")


if __name__ == "__main__":
    main()
