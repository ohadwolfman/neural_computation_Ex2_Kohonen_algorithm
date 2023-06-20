# neural_computation_Ex2_Kohonen_algorithm
In this project, we implemented the Kohonen (SOM) algorithm and explore its behavior under different scenarios. The Kohonen algorithm, also known as the Self-Organizing Map algorithm, is a type of artificial neural network used for unsupervised learning and data visualization.

# Part A:
The first part of the project involves fitting a line of random neurons in a data set consisting of points within the square range of  0 <=x<= 1 and 0 <=y<= 1. The distribution of the data points is uniform, while the Kohonen level is linearly ordered.The implementation of the algorithm will be tested with two different numbers of neurons: a small number (20) and a large number (200).
The aim is to observe the effect of varying the number of neurons on the algorithm's performance. Additionally, the evolution of the Kohonen map will be examined as the number of iterations increases.
The second part of the Part focuses on fitting a circle of neurons on a "donut" shape. The data set consists of points within the range 4<=x2+y2<=16. 
A circle of 300 neurons will be used for this task.

# Part B: (Did not implemented yet – to June 30th)
Part B of the project involves replicating an experiment known as the "monkey hand" problem. The data set is defined within a subset of the range 0 <=x<= 1 and 0<=y<=1, representing the shape of a hand. The Kohonen space consists of 400 neurons arranged in a 20 x 20 mesh. Initially, the entire hand is considered, and the evolution of the mesh is observed over iterations. Then, one of the fingers is "cut off," and the mesh is rearranged accordingly. The aim is to demonstrate how the Kohonen algorithm adapts to changes in the input data distribution.
