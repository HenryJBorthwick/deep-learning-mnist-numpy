from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data
import gzip, os

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 784 # Size of image vectors, each image is (28*28=783 pixels)
        self.num_classes = 10 # Number of classes/possible labels (digits 0-9) 
        self.batch_size = 100 # number of images per batch
        self.learning_rate = 0.5 # Step size for weight updates

        # TODO: Initialize weights and biases
        # MATRIX: 10 rows, one for each perceptron, 784 cols one for each pixel, SHAPE: (10, 784)
        # Start from 0, model knowns nothing to begin with
        self.W = np.zeros((self.num_classes, self.input_size)) # Weights: 10 perceptrons * 784 inputs
        # MATRIX: 10 rows, one for each perceptron, 1 col bias per perceptron, SHAPE: (10, 1)
        self.b = np.zeros((self.num_classes, 1)) # Biases: 10 perceptrons * 1

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: output, unscaled output values for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # Forward pass computes score for each perceptron for each image
        # input shape: (batch_size, 784), flattened pixel values normalized between 0 & 1
        # input shape: 2D array each row is an image (100 rows/images, each with 784 cols/pixels)
        # W shape: (10, 784)
        # b shape: (10, 1)
        # output shape: (batch_size, 10)

        # transpose math work, avoid looping over pixels
        # forward pass formula: f(x) = W * X + b
        output = np.dot(inputs, self.W.T) + self.b.T

        return output


    def back_propagation(self, inputs, outputs, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass. The learning algorithm for updating weights
        and biases is the Perceptron Learning Algorithm discussed in
        lecture (and described in the assignment writeup). This function should
        handle a batch_size number of inputs by taking the average of the gradients
        across all inputs in the batch.
        :param inputs: batch inputs (a batch of images)
        :param outputs: matrix that contains the unscaled output values of each
        class for each image
        :param labels: true labels
        :return: gradient for weights, and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        # HINT: np.argmax(outputs, axis=1) will give the index of the largest output

        # inputs SHAPE: (batch_size, 784)
        # output SHAPE: (batch_size, 10)
        # labels SHAPE: (batch_size,)

        # find predicted labels (highest scoring perceptron for each image)
        predicted = np.argmax(outputs, axis=1) # SHAPE: (batch_size, )

        # BACK PROPAGATION:
        # compute y values (how wrong each perceptron is) for each image
        batch_size = inputs.shape[0]
        y = np.zeros((batch_size, self.num_classes)) # SHAPE: (batch_size, 10), MATRIX: row is image, col, each entry is y value

        for i in range (batch_size):
            if predicted[i] != labels[i]: # if highest scoring perceptron does not match label
                y[i, labels[i]] = 1 # buff perceptron that should have won (perceptron that matches label)
                y[i, predicted[i]] = -1 # demote perceptron that did, but shouldn't win (perceptron with highest score and didn't match label)

        # compute gradient for weights and biases, average over batch
        gradW = np.dot(y.T, inputs) / batch_size # SHAPE: (10, batch_size) @ (batch_size, 784) -> (10, 784)
        gradB = np.mean(y, axis=0).reshape(self.num_classes, 1) #SHAPE: (10,) -> (10, 1)


    def accuracy(self, outputs, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param outputs: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy

    def gradient_descent(self, gradW, gradB):
        """
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        """
        # TODO: change the weights and biases of the model to descent the gradient

def train(model, train_inputs, train_labels):
    """
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    """

    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    for start in range(0, len(train_inputs), model.batch_size):
        inputs = train_inputs[start:start+model.batch_size]
        labels = train_labels[start:start+model.batch_size]

        # TODO: For every batch, compute then descend the gradients for the model's weights
        probabilities = model.call(inputs)
        gradientsW, gradientsB = model.back_propagation(inputs, probabilities, labels)
        model.gradient_descent(gradientsW, gradientsB)


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """

    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()


def main(mnist_data_folder):
    """
    Read in MNIST data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    """
    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels

    # TODO: Create Model

    # TODO: Train model by calling train() ONCE on all data

    # TODO: Test the accuracy by calling test() after running train()

    # TODO: Visualize the data by using visualize_results()

    print("end of assignment 1")


if __name__ == '__main__':
    #TODO: you might need to change this to something else if you run locally
    main("./MNIST_data")
