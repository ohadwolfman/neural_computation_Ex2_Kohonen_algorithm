# neural_computation_Ex2_Kohonen_algorithm
In this project, we implemented the Kohonen (SOM) algorithm and explore its behavior under different scenarios. The Kohonen algorithm, also known as the Self-Organizing Map algorithm, is a type of artificial neural network used for unsupervised learning and data visualization.

# Part A:
The first part of the project involves fitting a line of random neurons in a data set consisting of points within the square range of  0 <=x<= 1 and 0 <=y<= 1. The distribution of the data points is uniform, while the Kohonen level is linearly ordered. The implementation of the algorithm will be tested with two different numbers of neurons: a small number (20) and a large number (200).
The aim is to observe the effect of varying the number of neurons on the algorithm's performance. Additionally, the evolution of the Kohonen map will be examined as the number of iterations increases.
The second part of the Part focuses on fitting a circle of neurons on a "donut" shape. The data set consists of points within the range 4<=x2+y2<=16. 
A circle of 300 neurons will be used for this task.
![image](https://github.com/ohadwolfman/neural_computation_Ex2_Kohonen_algorithm/assets/98156296/751d767d-99ce-4db7-82d2-ada85180210e)
![image](https://github.com/ohadwolfman/neural_computation_Ex2_Kohonen_algorithm/assets/98156296/f16a74ef-b545-48db-acd8-efc5550bb798)



# Part B:
Part B of the project involves replicating an experiment known as the "monkey hand" problem. The data set is defined within a subset of the range 0 <=x<= 1 and 0<=y<=1, representing the shape of a hand. The Kohonen space consists of 400 neurons arranged in a 20 x 20 mesh. Initially, the entire hand is considered, and the evolution of the mesh is observed over iterations. Then, one of the fingers is "cut off," and the mesh is rearranged accordingly. The aim is to demonstrate how the Kohonen algorithm adapts to changes in the input data distribution.
Afterward we "cut off a finger" (i.e. the data points come only from the "hand with 4 fingers") and then continue from the stopping point in the previous section. 
We showed over snapshots of how the mesh is rearranged according to the smaller data points
![monkeyHand](https://github.com/ohadwolfman/neural_computation_Ex2_Kohonen_algorithm/assets/98156296/20442aed-161f-4bf9-a278-2d1a62052feb)
![full_hand_map_iterations](https://github.com/ohadwolfman/neural_computation_Ex2_Kohonen_algorithm/assets/98156296/7ed1eadf-c162-486b-ac26-a51204b32d38)
![FourFingersMonkeyHand](https://github.com/ohadwolfman/neural_computation_Ex2_Kohonen_algorithm/assets/98156296/00d1898f-99f8-452e-be7e-b8e368af453e)
![Plot2D_no_finger_iteration999](https://github.com/ohadwolfman/neural_computation_Ex2_Kohonen_algorithm/assets/98156296/2d607b83-2de0-4c2a-9885-ed9c54f9b9e2)


