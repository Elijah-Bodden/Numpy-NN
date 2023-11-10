import layer
import accessories
import dataLoader
import numpy as np


class Network:
    # Momentum booleans get overridden by non-None adaptive optimizer
    def __init__(
        self,
        layer_counts,
        init_function,
        activation_function,
        loss_function,
        learning_rate_function=lambda x: 0.001,
        out_activation_function=None,
        dropout=1,
        do_momentum=False,
        momentum_is_nesterov=False,
        decay_rate=0.9,
        second_rms_decay_rate=0.999,
        adaptive_type=None,
        RMS_epsilon=1e-8,
    ):
        """
        Initialize a dense network of layers with the uniform parameters (i.e. all layers have the same activation function, learning rate, etc. Not necessary but useful for demonstration purposes)

        Args:
        :param layer_counts list: a list of each layer's neuron count, including the input layer
        :param init_function function: initialization function overloaded to allow either one or two dimensions (bias vector length or weight matrix dimensions)
        :activation_function function: activation function for all layers except, optionally, the output layer
        :param loss_function function: loss function for the output layer
        :param learning_rate_function function: function accepting one parameter (the current timestep, equal to the total number of mini-batches performed so far) and returning a learning rate. Only used on learning-rate-based optimizers
        :param out_activation_function function: activation function for the output layer, defaults to activation_function
        :param dropout float: probability of keeping a unit during dropout (set to 1 for no dropout)
        :param do_momentum bool: whether to use momentum (only applies if adaptive_type isn't set)
        :param momentum_is_nesterov bool: whether to use nesterov acceleration if/when we do momentum
        :param decay_rate float: decay rate for momentum and RMSprop-like optimizers (equal to beta1 in adam)
        :param second_rms_decay_rate float: separate decay rate for the second RMS in adam-like optimizers (equal to beta2 in adam)
        :param adaptive_type string: preset optimizer type, overrides momentum-related parameters - one of [None, "adagrad", "adam", "rmsprop", or "adadelta"]
        :param RMS_epsilon float: epsilon value for RMS-like optimizers
        """
        if out_activation_function == None:
            out_activation_function = activation_function
        self.layers = [
            layer.Layer(
                layer_counts[i],
                layer_counts[i - 1],
                init_function,
                (
                    out_activation_function
                    if i == len(layer_counts)
                    else activation_function
                ),
                learning_rate_function,
                do_momentum,
                momentum_is_nesterov,
                decay_rate,
                second_rms_decay_rate,
                adaptive_type,
                RMS_epsilon,
            )
            for i in range(1, len(layer_counts))
        ]
        self.loss_function = loss_function
        self.dropout = dropout

    def forward_pass(self, input, is_inference, masks=[]):
        # is_inference determines dropout and whether to subtract momentum in nesterov layers, masks are dropout masks
        activations = [input]
        for idx, i in enumerate(self.layers):
            output = i.partial_forward_pass(activations[-1], is_inference)
            if not is_inference:
                activations.append(output * masks[idx] * (1.0 / self.dropout))
            else:
                activations.append(output)
        return activations

    def train(
        self,
        input,
        labels,
        do_test=True,
        test_data=None,
        test_labels=None,
        batch_size=8,
        epochs=1,
        norm_type=0,
        batch_norm=True,
        clip_at=1,
    ):
        """
        Train the network

        Args:
        :param input np.array: input data, formulated as an array of arrays of inputs
        :param labels np.array: labels for the input data, in the form of an array of arrays of target outputs
        :param do_test bool: whether to test the network after each epoch. Must also provide test_data and test_labels if True
        :param test_data np.array: input data for testing in between epochs. Required if do_test is True
        :param test_labels np.array: labels for the test data. Required if do_test is True
        :param batch_size int: mini-batch size
        :param epochs int: number of epochs to train for
        :param norm_type int: type of penalty term to use, L1 if 1, L2 if 2, otherwise no penalty
        :param batch_norm bool: whether to normalize each mini-batch to mean 0, std 1
        :param clip_at float: value to clip accumulated gradients at
        """

        for epoch in range(epochs):
            if do_test:
                self.test(test_data, test_labels)
            input, labels = accessories.shuffle(input, labels)
            for i in range(0, len(input) - batch_size, batch_size):
                if i % 10000 < batch_size:
                    print(f"Epoch {epoch+1}/{epochs}, sample {i-i%10000}/{len(input)}")
                dropout_masks = (
                    accessories.dropout_masks(
                        [len(l.biases) for l in self.layers[:-1]], self.dropout
                    )
                    + [np.ones((len(self.layers[-1].biases),))]
                    if self.dropout < 1
                    else [
                        np.ones(
                            len(l.biases),
                        )
                        for l in self.layers
                    ]
                )
                normalized = input[i : i + batch_size]
                if batch_norm:
                    normalized = accessories.batch_norm(input[i : i + batch_size])
                for idx in range(batch_size):
                    last_in_batch = idx == batch_size - 1
                    activations = self.forward_pass(
                        normalized[idx], False, masks=dropout_masks
                    )
                    current_deriv = self.loss_function(
                        activations[-1], labels[i + idx], True, norm_type
                    )
                    for j in range(len(self.layers) - 1, -1, -1):
                        current_deriv = self.layers[j].partial_gradient(
                            current_deriv,
                            activations[j + 1],
                            activations[j],
                            last_in_batch,
                            clip_at,
                        )

    def test(self, input, labels):
        if test_data is None or test_labels is None:
            return
        # Test the network on test data and labels
        correct = 0
        mean_loss = 0
        for i in range(len(input)):
            activations = self.forward_pass(input[i], True)[-1]
            # print(activations)
            if np.argmax(activations) == np.argmax(labels[i]):
                correct += 1
            mean_loss += self.loss_function(activations, labels[i], False) / len(input)
        print(f"Correct: {correct}/{len(input)}, mean loss: {mean_loss}")


net = Network(
    [784, 64, 64, 10],
    accessories.initializer,
    accessories.elu,
    accessories.MSE,
    out_activation_function=accessories.softmax,
    learning_rate_function=lambda x: 0.004,
    dropout=1,
    do_momentum=False,
    momentum_is_nesterov=False,
    decay_rate=0.95,
    second_rms_decay_rate=0.999,
    adaptive_type="adam",
    RMS_epsilon=1e-8,
)

data, labels = dataLoader.train_data()

test_data, test_labels = dataLoader.test_data()

net.train(
    data,
    labels,
    do_test=True,
    test_data=test_data,
    test_labels=test_labels,
    batch_size=8,
    epochs=3,
    norm_type=2,
    clip_at=1,
)

net.test(test_data, test_labels)