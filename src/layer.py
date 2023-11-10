import numpy as np
import accessories


class Layer:
    def __init__(
        self,
        self_neurons,
        previous_layer_neurons,
        initialization_function,
        activation_function,
        learning_rate_function,
        do_momentum,
        momentum_is_nesterov,
        decay_rate,
        second_rms_decay_rate,
        adaptive_type,
        RMS_epsilon,
    ):
        """
        Initialize a single dense layer for a neural network

        Args:
        :param self_neurons int: number of neurons in this layer
        :param previous_layer_neurons int: number of neurons in the layer to the left of this one (for initializing weights)
        :param initialization_function function: overloaded function accepting either bias vector length or weight matrix dimensions and returning an initialized tensor of this shape
        :param activation_function function: activation function for this layer
        :param learning_rate_function function: function accepting one parameter (the current timestep) and returning a learning rate
        :param do_momentum bool: whether to use momentum (only applies if adaptive_type isn't set)
        :param momentum_is_nesterov bool: whether to use nesterov acceleration if/when we do momentum (whether we subtract the momentum term from the weights when doing a non-inference forward pass)
        :param decay_rate float: decay rate for momentum and RMSprop-like optimizers (equal to beta1 in adam)
        :param second_rms_decay_rate float: separate decay rate for adam-like optimizers (beta2 in adam)
        :param adaptive_type string: preset optimizer type, overrides momentum-related parameters - one of [None, "adagrad", "adam", "rmsprop", or "adadelta"]
        :param RMS_epsilon float: epsilon value for RMS-like preset optimizers
        """
        self.biases = initialization_function(self_neurons)
        self.weights = initialization_function(previous_layer_neurons, self_neurons)
        self.activation_function = activation_function
        self.previous_layer_neurons = previous_layer_neurons
        self.learning_rate_function = learning_rate_function
        self.cached_weights = np.zeros((len(self.weights), len(self.weights[0])))
        self.cached_biases = np.zeros((len(self.biases),))
        # Used for momentum and adaptive optimizers like adagrad and adam
        self.weight_accumulator = np.zeros((len(self.weights), len(self.weights[0])))
        self.bias_accumulator = np.zeros((len(self.biases),))
        # Used in adaptive optimizers like adadelta and adam which require a second accumulator
        self.second_weight_accumulator = np.zeros(
            (len(self.weights), len(self.weights[0]))
        )
        self.second_bias_accumulator = np.zeros((len(self.biases),))
        self.do_momentum = do_momentum and not adaptive_type
        # Used for momentum and RMSprop-like decaying optimizers
        self.decay_rate = decay_rate
        # Used adam-like optimizers with a separate decay rate for the second accumulator
        self.second_rms_decay_rate = second_rms_decay_rate
        self.momentum_is_nesterov = momentum_is_nesterov and not adaptive_type
        if adaptive_type not in [None, "adagrad", "adam", "rmsprop", "adadelta"]:
            raise ValueError(f"Invalid adaptive_type: {adaptive_type}")
        self.adaptive_type = adaptive_type
        self.RMS_epsilon = RMS_epsilon
        self.timestep = 0

    def partial_forward_pass(self, prev_activations, is_inference):
        # Perform this layer's step in a forward pass. is_inference determines whether to subtract momentum in nesterov layers
        if is_inference or not self.momentum_is_nesterov:
            return self.activation_function(
                self.weights @ prev_activations + self.biases
            )
        return self.activation_function(
            (self.weights - self.weight_accumulator) @ prev_activations
            + self.biases
            - self.bias_accumulator
        )

    def partial_gradient(
        self, prev_step, rightActivations, leftActivations, save_grad, clip_at
    ):
        # Compute gradients of a layer given its activations and the activations of the layer to its left and update parameters. save_grad tells whether this is the last step in the batch and thus the accumulated gradients should be applied. The chained conditionals determine the update function based on the optimizer type
        bias_grad, weight_grad, this_step = accessories.compute_gradient(
            prev_step, rightActivations, leftActivations, self
        )
        self.cached_weights += weight_grad
        self.cached_biases += bias_grad
        if save_grad:
            # For some reason, every optimizer except vanilla SGD prefers learning rate multiplied after clipping, but vanilla SGD does better with it done before
            if self.adaptive_type == None and not self.do_momentum:
                self.cached_weights *= self.learning_rate_function(self.timestep)
                self.cached_biases *= self.learning_rate_function(self.timestep)
            weight_norm = np.sum(self.cached_weights**2)
            if weight_norm > clip_at**2:
                self.cached_weights *= (clip_at**2 / weight_norm) ** 0.5
            # Optimizer-specific parameter update functions
            self.timestep += 1
            if self.do_momentum:
                self.weight_accumulator = accessories.decaying_mean(
                    self.decay_rate,
                    self.weight_accumulator,
                    self.cached_weights * self.learning_rate_function(self.timestep),
                )
                self.bias_accumulator = accessories.decaying_mean(
                    self.decay_rate,
                    self.bias_accumulator,
                    self.cached_biases * self.learning_rate_function(self.timestep),
                )
                self.weights -= self.weight_accumulator
                self.biases -= self.bias_accumulator
            elif self.adaptive_type == None:
                self.weights -= self.cached_weights
                self.biases -= self.cached_biases
            elif self.adaptive_type == "adagrad":
                self.weight_accumulator += self.cached_weights**2
                self.bias_accumulator += self.cached_biases**2
                self.weights -= (
                    self.learning_rate_function(self.timestep)
                    / np.sqrt(self.weight_accumulator + self.RMS_epsilon)
                    * self.cached_weights
                )
                self.biases -= (
                    self.learning_rate_function(self.timestep)
                    / np.sqrt(self.bias_accumulator + self.RMS_epsilon)
                    * self.cached_biases
                )
            elif self.adaptive_type == "rmsprop":
                self.weight_accumulator = accessories.decaying_mean_squared(
                    self.decay_rate, self.weight_accumulator, self.cached_weights
                )
                self.bias_accumulator = accessories.decaying_mean_squared(
                    self.decay_rate, self.bias_accumulator, self.cached_biases
                )
                self.weights -= (
                    self.learning_rate_function(self.timestep)
                    / np.sqrt(self.weight_accumulator + self.RMS_epsilon)
                ) * self.cached_weights
                self.biases -= (
                    self.learning_rate_function(self.timestep)
                    / np.sqrt(self.bias_accumulator + self.RMS_epsilon)
                ) * self.cached_biases
            elif self.adaptive_type == "adadelta":
                # Adaptive gradient-scaled denominator:
                self.weight_accumulator = accessories.decaying_mean_squared(
                    self.decay_rate, self.weight_accumulator, self.cached_weights
                )
                self.bias_accumulator = accessories.decaying_mean_squared(
                    self.decay_rate, self.bias_accumulator, self.cached_biases
                )
                weight_update = (
                    np.sqrt(self.second_weight_accumulator + self.RMS_epsilon)
                    / np.sqrt(self.weight_accumulator + self.RMS_epsilon)
                ) * self.cached_weights
                bias_update = (
                    np.sqrt(self.second_bias_accumulator + self.RMS_epsilon)
                    / np.sqrt(self.bias_accumulator + self.RMS_epsilon)
                ) * self.cached_biases
                self.weights -= weight_update
                self.biases -= selfbias_update
                # Adaptive update-scaled numerator:
                self.second_weight_accumulator = accessories.decaying_mean_squared(
                    self.decay_rate,
                    self.second_weight_accumulator,
                    weight_update,
                )
                self.second_bias_accumulator = accessories.decaying_mean_squared(
                    self.decay_rate,
                    self.second_bias_accumulator,
                    bias_update,
                )
            elif self.adaptive_type == "adam":
                # Momentum-ed gradients
                self.weight_accumulator = accessories.decaying_mean(
                    self.decay_rate,
                    self.weight_accumulator,
                    self.cached_weights,
                )
                self.bias_accumulator = accessories.decaying_mean(
                    self.decay_rate,
                    self.bias_accumulator,
                    self.cached_biases,
                )
                # RMS adaptive learning rate
                self.second_weight_accumulator = accessories.decaying_mean_squared(
                    self.second_rms_decay_rate,
                    self.second_weight_accumulator,
                    self.cached_weights,
                )
                self.second_bias_accumulator = accessories.decaying_mean_squared(
                    self.second_rms_decay_rate,
                    self.second_bias_accumulator,
                    self.cached_biases,
                )
                # Bias-correct accumulators and update parameters
                self.weights -= (
                    (
                        self.learning_rate_function(self.timestep)
                        / np.sqrt(
                            self.second_weight_accumulator
                            / (1 - self.second_rms_decay_rate**self.timestep)
                            + self.RMS_epsilon
                        )
                    )
                    * self.weight_accumulator
                    / (1 - self.decay_rate**self.timestep)
                )
                self.biases -= (
                    (
                        self.learning_rate_function(self.timestep)
                        / np.sqrt(
                            self.second_bias_accumulator
                            / (1 - self.second_rms_decay_rate**self.timestep)
                            + self.RMS_epsilon
                        )
                    )
                    * self.bias_accumulator
                    / (1 - self.decay_rate**self.timestep)
                )
            self.cached_weights = np.zeros((len(self.weights), len(self.weights[0])))
            self.cached_biases = np.zeros((len(self.biases),))
        return this_step
