import gzip
import numpy as np

def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels). Read the data of the file into a buffer and use
    np.frombuffer to turn the data into a NumPy array. Keep in mind that
    each file has a header of a certain size. This method should be called
    within the main function of the assignment.py file to get BOTH the train and
    test data. If you change this method and/or write up separate methods for
    both train and test data, we will deduct points.

    Hint: look at the writeup for sample code on using the gzip library

    :param inputs_file_path: file path for inputs, something like
    'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, something like
    'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs as float32 and labels as int8
    """

    # Load and unzip inputs
    with gzip.open(inputs_file_path, 'rb') as f:
        # skip header bytes for images
        f.read(16)
        # read the buffer of bytes (num_examples * 28 * 28) for images
        buffer = f.read(num_examples * 28 * 28)
        # convert buffer of bytes to NumPy array
        inputs = np.frombuffer(buffer, dtype=np.uint8)
        # reshape NumPy Array
        inputs = np.reshape(inputs, (num_examples, 28, 28))

    # Load and unzip labels
    with gzip.open(labels_file_path, 'rb') as f:
        # skip header bytes for labels
        f.read(8)
        # read the buffer of bytes (num_examples) for labels
        buffer = f.read(num_examples)
        # convert buffer to NumPy array
        labels = np.frombuffer(buffer, dtype=np.uint8)
    
    # Normalize inputs to ranges between 0 to 1
    # first covert to float32 to do this
    inputs = inputs.astype(np.float32)

    # now normalize by dividing each pixel by 255, each pixel is exactly 1 byte
    inputs = inputs / 255.0

    return inputs, labels
