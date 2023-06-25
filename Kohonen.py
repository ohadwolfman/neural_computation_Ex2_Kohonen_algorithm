#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Kohonen:
    def __init__(self, neurons_amount=20, learning_rate=0.7, radius=1.0):
        self.neurons_amount = neurons_amount
        self.learning_rate = learning_rate
        self.radius = radius
        self.weights = None

    def initialize_weights(self, input_dim):
        self.weights = np.random.rand(self.neurons_amount, input_dim)

    def find_winner(self, input_data):
        distances = np.linalg.norm(input_data - self.weights, axis=1)
        return np.argmin(distances)

    def update_weights(self, input_data, winner_idx):
        distance_to_winner = np.abs(np.arange(self.neurons_amount) - winner_idx)
        influence = np.exp(-distance_to_winner / self.radius)
        self.weights += self.learning_rate * influence[:, np.newaxis] * (input_data - self.weights)

    def fit(self, data, num_epochs=1000):
        input_dim = data.shape[1]
        self.initialize_weights(input_dim)

        for epoch in range(num_epochs):
            np.random.shuffle(data)

            for input_data in data:
                winner_idx = self.find_winner(input_data)
                self.update_weights(input_data, winner_idx)

        return self.weights


def create_circle_dataset():
    dataset = []
    while len(dataset) < 1000:  # Generate 100 points
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        distance_squared = x ** 2 + y ** 2
        if 4 <= distance_squared <= 16:
            dataset.append([x, y])
    return np.array(dataset)


def create_plot(dataset):
    # Extract x and y coordinates from the dataset
    x_coords = dataset[:, 0]
    y_coords = dataset[:, 1]

    # Plot the dataset
    plt.scatter(x_coords, y_coords)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dataset')
    plt.grid(True)
    plt.show()

def squareDataSet():
    # Set the parameters
    input_dim = 2  # Input data dimension
    learning_rate = 0.5  # Learning rate
    num_epochs = 1000  # Number of training epochs

    # Generate the input data
    np.random.seed(0)
    data = np.random.rand(1000, input_dim)  # Random data points between 0 and 1

    # Create a Kohonen network with 20 neurons
    kohonen_20 = Kohonen(neurons_amount=20, learning_rate=learning_rate)
    weights_20 = kohonen_20.fit(data, num_epochs)

    # Create a Kohonen network with 200 neurons
    kohonen_200 = Kohonen(neurons_amount=200, learning_rate=learning_rate)
    weights_200 = kohonen_200.fit(data, num_epochs)

    # Plot the final SOM for 20 neurons
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(weights_20[:, 0], weights_20[:, 1], c='r')
    plt.title('Kohonen Self-Organizing Map (20 Neurons)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Plot the final SOM for 200 neurons
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(weights_200[:, 0], weights_200[:, 1], c='r')
    plt.title('Kohonen Self-Organizing Map (200 Neurons)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def circleDataSet():
    data = create_circle_dataset()
    create_plot(data)

    # Set the parameters
    input_dim = 2  # Input data dimension
    output_dim = 20  # Output map dimension
    learning_rate = 0.5  # Learning rate
    num_epochs = 1000  # Number of training epochs

    # Create a Kohonen network with 20 neurons
    kohonen_20 = Kohonen(neurons_amount=output_dim, learning_rate=learning_rate)
    weights_20 = kohonen_20.fit(data, num_epochs)

    # Create a Kohonen network with 200 neurons
    kohonen_200 = Kohonen(neurons_amount=200, learning_rate=learning_rate)
    weights_200 = kohonen_200.fit(data, num_epochs)

    # Plot the final SOM for 20 neurons
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(weights_20[:, 0], weights_20[:, 1], c='r')
    plt.title('Kohonen Self-Organizing Map (20 Neurons)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Plot the final SOM for 200 neurons
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(weights_200[:, 0], weights_200[:, 1], c='r')
    plt.title('Kohonen Self-Organizing Map (200 Neurons)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def monkeyHand():
    # Set the parameters
    hand = cv2.imread("images/monkeyHand.jpg")
    hand = cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)
    hand = cv2.resize(hand,(0,0), fx=0.5,fy=0.5)
    points = np.argwhere(hand != 255).astype(np.float32)
    plt.imshow(hand)
    plt.show()
    print(points)

    # input_dim = 2  # Input data dimension
    # learning_rate = 0.5  # Learning rate
    # num_epochs = 1000  # Number of training epochs
    #
    # # Generate the input data
    # np.random.seed(0)
    # data = np.random.rand(1000, input_dim)  # Random data points between 0 and 1
    #
    # # Create a Kohonen network with 300 neurons
    # kohonen_20 = Kohonen(neurons_amount=300, learning_rate=learning_rate)
    # weights_20 = kohonen_20.fit(data, num_epochs)
    #
    # # Plot the final SOM for 20 neurons
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.scatter(weights_20[:, 0], weights_20[:, 1], c='r')
    # plt.title('Kohonen Self-Organizing Map (20 Neurons)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()


def main():
    # -------------- Part A ---------------
    # --------------Square shape---------------
    # squareDataSet()

    # --------------Circle shape---------------
    # circleDataSet()

    # -------------- Part B ---------------
    monkeyHand()


if __name__ == '__main__':
    main()