import mnist
import numpy as np


def unpack_image(image):
    # Unwrap an image into 1d by stacking rows end-to-end
    j = []
    for i in image:
        j += i.tolist()
    return np.array(j)


def int_to_one_hot(tensor):
    # Convert a tensor of integers 0-9 to a tensor of one-hots
    return np.array(
        [np.array([1 if i == int else 0 for i in range(10)]) for int in tensor]
    )


def train_data():
    # Load, normalize, and unwrap training data, convert labels to one-hot
    data = np.array([unpack_image(x / 255) for x in list(mnist.train_images())])
    data_labels = int_to_one_hot(mnist.train_labels())
    return data, data_labels


def test_data():
    # Do the same thing but on the test data
    test_data = np.array([unpack_image(x / 255) for x in list(mnist.test_images())])
    test_labels = int_to_one_hot(mnist.test_labels())
    return test_data, test_labels