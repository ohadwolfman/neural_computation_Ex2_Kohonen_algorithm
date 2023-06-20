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

    def fit(self, data, num_epochs=1000, toPlt=False):
        input_dim = data.shape[1]
        self.initialize_weights(input_dim)
        if toPlt:
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for epoch in range(num_epochs):
            np.random.shuffle(data)

            if toPlt:
                if epoch == 0:
                    plot_res_return(axes, 0, 0, data, self.weights)
                    axes[0, 0].set_title('0 iterations')
                if epoch == 1:
                    plot_res_return(axes, 0, 1, data, self.weights)
                    axes[0, 1].set_title('1 iterations')
                if epoch == 30:
                    plot_res_return(axes, 0, 2, data, self.weights)
                    axes[0, 2].set_title('30 iterations')
                if epoch == 70:
                    plot_res_return(axes, 1, 0, data, self.weights)
                    axes[1, 0].set_title('70 iterations')
                if epoch == 100:
                    plot_res_return(axes, 1, 1, data, self.weights)
                    axes[1, 1].set_title('100 iterations')
                if epoch == 300:
                    plot_res_return(axes, 1, 2, data, self.weights)
                    axes[1, 2].set_title('300 iterations')
                if epoch == 600:
                    plot_res_return(axes, 2, 0, data, self.weights)
                    axes[2, 0].set_title('600 iterations')
                if epoch == 900:
                    plot_res_return(axes, 2, 1, data, self.weights)
                    axes[2, 1].set_title('900 iterations')

            for input_data in data:
                winner_idx = self.find_winner(input_data)
                self.update_weights(input_data, winner_idx)

        return self.weights


def Create_Kohonen_network(data, output_dim, learning_rate, num_epochs, toPlt):
    kohonen = Kohonen(neurons_amount=output_dim, learning_rate=learning_rate)
    weights = kohonen.fit(data, num_epochs, toPlt)
    plot_res(data, weights)


# plotting the final grid
def plot_res(data, weights):
    # Plot the final SOM for 20 neurons
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(weights[:, 0], weights[:, 1], c='r')
    plt.title('1000 iterations')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# for creating a 3x3 fig
def plot_res_return(axes, raw, col, data, weights):
    # Plot the final SOM for 20 neurons
    axes[raw, col].scatter(data[:, 0], data[:, 1])
    axes[raw, col].scatter(weights[:, 0], weights[:, 1], c='r')


# showing the data on a plt
def create_data_plot(dataset):
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


# creating the square non-uniform dataset
def create_random_dataset(num_points):
    """
    Create a dataset of points with random values between 0 and 1.
    :param num_points: Number of points to generate.
    :return: Numpy array of shape (num_points, 2) representing the dataset.
    """

    dataset = np.random.rand(num_points, 2)
    return dataset


# creating the circle (donut) non-uniform dataset
def create_circle_dataset():
    dataset = []

    while len(dataset) < 1000:  # Generate 100 points
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        distance_squared = x ** 2 + y ** 2
        if 4 <= distance_squared <= 16:
            dataset.append([x, y])
    return np.array(dataset)


# creating the first non-uniform dataset on a square -
# In this code, the data array is generated with non-uniform distributions based on 
# the likelihood proportional to the size of x and uniform to the size of y.
# The x values are generated uniformly between 0 and 1, and the y values are generated 
# uniformly between 0 and the corresponding x value.
def create_unUni_1_dataset():
    # Generate the input data with non-uniform distributions

    data_unUni_1 = np.zeros((1000, 2))

    # Generate x values with likelihood proportional to the size of x
    data_unUni_1[:, 0] = np.random.uniform(0, 1, 1000)
    data_unUni_1[:, 1] = np.random.uniform(0, data_unUni_1[:, 0], 1000)
    return data_unUni_1


# creating the first non-uniform dataset on a circle -
# The resulting dataset will consist of 1000 data points that are
# distributed in a circular pattern within the specified radius range.
# The points will be closer to the origin and sparser as the distance from the origin increases.
def create_unUni_1_circle_dataset():
    dataset = []
    while len(dataset) < 1000:  # Generate 1000 points
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, x)
        distance_squared = x ** 2 + y ** 2
        if 4 <= distance_squared <= 16:
            dataset.append([x, y])
    return np.array(dataset)


# creating the second non-uniform dataset on a square -
def create_unUni_2_dataset(num_points):
    dataset = []
    for _ in range(num_points):
        rand = np.random.random()
        if rand < 0.4:
            x = np.random.uniform(0, 0.5)
            y = np.random.uniform(0, 0.5)
        elif rand < 0.8:
            x = np.random.uniform(0.5, 1)
            y = np.random.uniform(0.5, 1)
        else:
            if np.random.random() < 0.5:
                x = np.random.uniform(0, 0.5)
                y = np.random.uniform(0.5, 1)
            else:
                x = np.random.uniform(0.5, 1)
                y = np.random.uniform(0, 0.5)
        dataset.append((x, y))
    return np.array(dataset)


# creating the second non-uniform dataset on a circle (donut) -
def create_unUni_2_circle_dataset(num_points):
    dataset = []

    for _ in range(num_points):
        r = np.random.uniform(np.sqrt(4), np.sqrt(16))  # Radius range: [2, 4)
        theta = np.random.uniform(0, 2 * np.pi)  # Angle range: [0, 2pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        if x <= 0 and y >= 0:  # First quarter
            probability = 1 / 5
        elif x >= 0 and y <= 0:  # Third quarter
            probability = 1 / 5
        else:
            probability = 4 / 5

        if np.random.uniform(0, 1) < probability:
            dataset.append((x, y))
    return np.array(dataset)


def main():
    # creating a square uniform dataset
    data_square = create_random_dataset(1000)
    create_data_plot(data_square)

    # activate the kohonen algorithm with 20 neurons and plotting the results
    Create_Kohonen_network(data_square, output_dim=20, learning_rate=0.5, num_epochs=1000, toPlt=False)

    # activate the kohonen algorithm with 20 neurons and plotting the results
    Create_Kohonen_network(data_square, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=False)

    # creating the first square un-uniform dataset
    data_unUni_1 = create_unUni_1_dataset()
    create_data_plot(data_unUni_1)
    # same as above
    Create_Kohonen_network(data_unUni_1, output_dim=20, learning_rate=0.5, num_epochs=1000, toPlt=False)
    Create_Kohonen_network(data_unUni_1, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=False)

    # creating the second square un-uniform dataset
    data_unUni_2 = create_unUni_2_dataset(1000)
    create_data_plot(data_unUni_2)
    Create_Kohonen_network(data_unUni_2, output_dim=20, learning_rate=0.5, num_epochs=1000, toPlt=False)
    Create_Kohonen_network(data_unUni_2, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=False)

    # showing a fig containing 9 plots from 9 steps in the kohonen algorithm-
    # 1.the regular square uniform data
    Create_Kohonen_network(data_square, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=True)
    # 2.the second un-uniform square data
    Create_Kohonen_network(data_unUni_2, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=True)

    # creating a donut uniform dataset
    data_donut = create_circle_dataset()
    create_data_plot(data_donut)
    # activate the kohonen algorithm with 20 neurons and plotting the results
    Create_Kohonen_network(data_donut, output_dim=20, learning_rate=0.5, num_epochs=1000, toPlt=False)
    # activate the kohonen algorithm with 20 neurons and plotting the results
    Create_Kohonen_network(data_donut, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=False)

    # creating the first donut un-uniform dataset
    data_donut_unUni_1 = create_unUni_1_circle_dataset()
    create_data_plot(data_donut_unUni_1)
    # same as above
    Create_Kohonen_network(data_donut_unUni_1, output_dim=20, learning_rate=0.5, num_epochs=1000, toPlt=False)
    Create_Kohonen_network(data_donut_unUni_1, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=False)

    # creating the second donut un-uniform dataset
    data_donut_unUni_2 = create_unUni_2_circle_dataset(1000)
    create_data_plot(data_donut_unUni_2)
    Create_Kohonen_network(data_donut_unUni_2, output_dim=20, learning_rate=0.5, num_epochs=1000, toPlt=False)
    Create_Kohonen_network(data_donut_unUni_2, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=False)

    # showing a fig containing 9 plots from 9 steps in the kohonen algorithm-
    # 1.the regular donut uniform data
    Create_Kohonen_network(data_donut, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=True)
    # 2.the second un-uniform square data
    Create_Kohonen_network(data_donut_unUni_2, output_dim=200, learning_rate=0.5, num_epochs=1000, toPlt=True)


if __name__ == "__main__":
    main()
