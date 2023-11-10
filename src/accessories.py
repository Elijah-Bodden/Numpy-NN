from multipledispatch import dispatch
import numpy as np


@dispatch(int)
def initializer(size1):
    # Initialize biases to 0
    return np.zeros((size1,))


@dispatch(int, int)
def initializer(size1, size2):
    # Glorot initialize weights
    range = np.sqrt(6 / (size1 + size2))
    return np.random.uniform(-range, range, (size2, size1))


def sigmoid(vector, derivative=False):
    # Sigmoid activation
    sig = 1 / (1 + np.exp(-vector))
    if not derivative:
        return sig
    else:
        return sig * (1 - sig)


def MSE(vector, target, derivative=False, norm=0, penalty_lambda=0.01):
    # Mean squared loss
    if derivative:
        return 2 * (vector - target) + penalty_term(vector, norm, True, penalty_lambda)
    else:
        return np.sum((vector - target) ** 2) + penalty_term(
            vector, norm, False, penalty_lambda
        )


def penalty_term(vector, norm, derivative=False, lambda_=0.01):
    # Compute penalty term for regularization (no penalty if norm is not 1 or 2)
    if not 3 > norm >= 0:
        raise ValueError(f"Invalid norm kind: {norm}")
    if not derivative:
        match norm:
            case 1:
                return lambda_ * np.sum(np.abs(vector))
            case 2:
                return lambda_ * 0.5 * np.sum(vector**2)
            case _:
                return 0
    else:
        match norm:
            case 1:
                return lambda_ * np.sign(vector)
            case 2:
                return lambda_ * vector
            case _:
                return 0


def compute_gradient(
    prev_step, rightActivations, leftActivations, this_layer, clip_value=1
):
    # Compute gradients of a layer given its activations and the activations of the layer to its left
    recovered_mask = np.where(rightActivations == 0, 0, 1)
    # Clip the gradient to prevent explosion
    norm_sq = np.sum(prev_step**2)
    if norm_sq > clip_value**2:
        prev_step *= (clip_value**2 / norm_sq) ** 0.5
    this_step = prev_step * this_layer.activation_function(rightActivations, True)
    bias_grad = this_step * recovered_mask
    weight_grad = np.outer(this_step, leftActivations)
    # Backpropagate the gradient to create the next layer's prev_step
    this_step = this_layer.weights.T @ this_step
    return bias_grad, weight_grad, this_step


def elu(vector, derivative=False):
    # ELU activation
    if not derivative:
        return np.where(vector > 0, vector, np.exp(vector) - 1)
    else:
        return np.where(vector > 0, 1, np.exp(vector))


def softmax(vector, derivative=False):
    if not derivative:
        return np.exp(vector) / np.sum(np.exp(vector))
    else:
        return np.diag(softmax(vector, derivative=False)) - np.outer(
            softmax(vector, derivative=False), softmax(vector, derivative=False)
        )


def cross_entropy(
    vector, target, derivative=False, norm=0, penalty_lambda=0.01, clipEpsilon=1e-10
):
    # Cross entropy loss
    vector = np.clip(vector, clipEpsilon, 1 - clipEpsilon)
    if not derivative:
        return -np.sum(target * np.log(vector)) + penalty_term(
            vector, norm, False, penalty_lambda
        )
    else:
        return vector - target + penalty_term(vector, norm, True, penalty_lambda)


def batch_norm(batch):
    # Normalize a batch to mean 0, std 1
    return (batch - np.mean(batch)) / np.std(batch)


def dropout_masks(lengths, prob):
    # Generate dropotut masks for a batch with prob% probability of keeping each unit
    return [
        np.random.choice([0, 1], size=(length,), p=[1 - prob, prob])
        for length in lengths
    ]


def shuffle(data, labels):
    # Shuffle data and labels with the same pattern
    idx = np.random.permutation(len(data))
    return data[idx], labels[idx]


def decaying_mean(decay_rate, mean, vector, squared=True):
    # Compute one step of the vector exponentially decaying mean for momentum and adaptive optimizers
    return decay_rate * mean + (1 - decay_rate) * vector


def decaying_mean_squared(decay_rate, mean, vector, squared=True):
    # Do the same thing but on the squares of the vector
    return decay_rate * mean + (1 - decay_rate) * vector**2