import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    """Vanilla Gradient Descent."""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, parameters, gradients):
        for (w, b), (dw, db) in zip(parameters, gradients):
            w -= self.lr * dw
            b -= self.lr * db

class MomentumGD:
    """Gradient Descent with Momentum."""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, parameters, gradients):
        if self.velocity is None:
            self.velocity = [(np.zeros_like(w), np.zeros_like(b)) for w, b in parameters]

        for i, ((w, b), (dw, db)) in enumerate(zip(parameters, gradients)):
            self.velocity[i] = (self.momentum * self.velocity[i][0] - self.lr * dw,
                                self.momentum * self.velocity[i][1] - self.lr * db)
            w += self.velocity[i][0]
            b += self.velocity[i][1]

class NAG:
    """Nesterov Accelerated Gradient (NAG)."""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, parameters, gradients):
        if self.velocity is None:
            self.velocity = [(np.zeros_like(w), np.zeros_like(b)) for w, b in parameters]

        for i, ((w, b), (dw, db)) in enumerate(zip(parameters, gradients)):
            lookahead_w = w + self.momentum * self.velocity[i][0]
            lookahead_b = b + self.momentum * self.velocity[i][1]

            self.velocity[i] = (self.momentum * self.velocity[i][0] - self.lr * dw,
                                self.momentum * self.velocity[i][1] - self.lr * db)

            w += self.velocity[i][0]
            b += self.velocity[i][1]

# Visualization comparison
def simulate_training(optimizer, epochs=50):
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X @ np.array([[2], [-1], [1]]) + 0.5  # True weights
    w = np.random.randn(3, 1)
    b = np.random.randn(1)
    losses = []

    for _ in range(epochs):
        y_pred = X @ w + b
        loss = np.mean((y - y_pred) ** 2)
        dw = -2 * X.T @ (y - y_pred) / len(X)
        db = -2 * np.mean(y - y_pred)
        optimizer.update([(w, b)], [(dw, db)])
        losses.append(loss)

    return losses

if __name__ == "__main__":
    vanilla = GradientDescent(learning_rate=0.05)
    momentum = MomentumGD(learning_rate=0.05, momentum=0.9)
    nag = NAG(learning_rate=0.05, momentum=0.9)

    loss_vanilla = simulate_training(vanilla)
    loss_momentum = simulate_training(momentum)
    loss_nag = simulate_training(nag)

    # Plot all three
    plt.figure(figsize=(10, 6))
    plt.plot(loss_vanilla, label="Vanilla GD")
    plt.plot(loss_momentum, label="Momentum GD")
    plt.plot(loss_nag, label="NAG")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Comparison of Gradient Descent Variants")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("project/gd_comparison_plot.png")
    plt.show()
