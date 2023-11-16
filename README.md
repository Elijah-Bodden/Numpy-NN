# Numpy-NN
Neural nets from scratch in numpy with:
- MSE/cross-entropy loss, ELU/Sigmoid/Softmax activation functions, Xavier initialization
- Optional L1/2 regularization
- Batch normalization
- Gradient clipping
- Dropout
- Nesterov and standard momentum
- Optional learning-rate scheduling
- Adam, RMSprop, Adagrad, and Adadelta optimizers
- And more

## Hyperparameters and implementation details
The project is implemented in three parts: Network, Layer, and Accessories
### Network
The network class creates a dense, homogenous MLP and is mainly a dummy example of a network built using the real star of the show - the layer class. The following, taken from docstrings in the code, describe the two most complex elements, the constructor and the train function. The other, less complex functions documented with light comments in the code:  

`__init__`:
Initialize a dense network of layers with the uniform parameters (i.e. all layers have the same activation function, learning rate, etc. Not necessary but useful for demonstration purposes)

Args:

- `layer_counts:list` - a list of each layer's neuron count, including the input layer
- `init_function:function` - initialization function overloaded to allow either one or two dimensions (bias vector length or weight matrix dimensions)
- `activation_function:function` - activation function for all layers except, optionally, the output layer
- `loss_function:function` - loss function for the output layer
- `learning_rate_function:function` - function accepting one parameter (the current timestep, equal to the total number of mini-batches performed so far) and returning a learning rate. Only used on learning-rate-based optimizers
- `out_activation_function:function` - activation function for the output layer, defaults to activationFunction
- `dropout:float` - probability of keeping a unit during dropout (set to 1 for no dropout)
- `do_momentum:bool` - whether to use momentum (only applies if adaptive_type isn't set)
- `momentum_is_nesterov:bool` - whether to use nesterov acceleration if/when we do momentum
- `decay_rate:float` - decay rate for momentum and RMSprop-like optimizers (equal to beta1 in adam)
- `second_rms_decay_rate:float` - separate decay rate for the second RMS in adam-like optimizers (equal to beta2 in adam)
- `adaptive_type:string` - preset optimizer type, overrides momentum-related parameters - one of [None, "adagrad", "adam", "rmsprop", or "adadelta"]
- `RMS_epsilon:float` - epsilon value for RMS-like optimizers

`train`:
Train the network

Args:

- `input:np.array` - input data, formulated as an array of arrays of inputs
- `labels:np.array` - labels for the input data, in the form of an array of arrays of target outputs
- `do_test:bool` - whether to test the network after each epoch. Must also provide test_data and test_labels if True
- `test_data:np.array` - input data for testing in between epochs. Required if do_test is True
- `test_labels:np.array` - labels for the test data. Required if do_test is True
- `batch_size:int` - mini-batch size
- `epochs:int` - number of epochs to train for
- `norm_type:int` - type of penalty term to use, L1 if 1, L2 if 2, otherwise no penalty
- `batch_norm:bool` - whether to normalize each mini-batch to mean 0, std 1
- `clip_at:float` - value to clip accumulated gradients at
### Layer
The layer class is an extensible representation of a network layer, plugging in components from the `accessories` module to change its behavior. In principle, many more kinds of network than the simple dense MLP implemented in `network` can be built with this class.

`__init__`
Initialize a single layer for a neural network

Args:

- `self_neurons:int` - number of neurons in this layer
- `previous_layer_neurons:int` - number of neurons in the layer to the left of this one (for initializing weights)
- `initialization_function:function` - overloaded:function accepting either bias vector length or weight matrix dimensions and returning an initialized tensor of this shape
- `activation_function:function` - activation:function for this layer
- `learning_rate_function:function` -:function accepting one parameter (the current timestep) and returning a learning rate
- `do_momentum:bool` - whether to use momentum (only applies if adaptive_type isn't set)
- `momentum_is_nesterov:bool` - whether to use nesterov acceleration if/when we do momentum (whether we subtract the momentum term from the weights when doing a non-inference forward pass)
- `decay_rate:float` - decay rate for momentum and RMSprop-like optimizers (equal to beta in adam and adadelta)
- `second_rms_decay_rate:float` - separate decay rate for the second RMS in adadelta and adam
- `adaptive_type:string` - preset optimizer type, overrides momentum-related parameters - one of [None, "adagrad", "adam", "rmsprop", or "adadelta"]
- `RMS_epsilon:float` - epsilon value for RMS-like preset optimizers
### Accessories
The accessories class contains pluggable, modifiable functions like the partial backpropagation algorithm, initializers, utilities, and activation functions that can be passed to the Network and Layer classes to customize their behavior.

## Data
The example net trains on the MNIST dataset loaded with `dataLoader.py`, but any two numpy arrays - one full of flattened input activations and one of the target outputs - will work to train the network class.
