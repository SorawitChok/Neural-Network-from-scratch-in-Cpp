# Neural Network from scratch in C++

Welcome to the **Neural Network from Scratch in C++** project! This repository features a straightforward implementation of a neural network built entirely from the ground up using C++. Designed to engage AI and machine learning enthusiasts, this project provides a hands-on opportunity to explore the mathematical and programming principles behind neural networks. Whether you're a learner or an experienced developer, you'll gain deeper insights into the inner workings of neural networks and their underlying algorithms.

## Table of Contents
- [Concept & Intuition](#Concept-&-Intuition)
- [Underlying Mathematics](#Underlying-Mathematics)
  - [Forward Propagation](#Forward-Propagation)
  - [Gradient Descent](#Gradient-Descent)
  - [Backward Propagation](#Backward-Propagation)
- [Breaking into Modules](#Breaking-into-Modules)

## Concept & Intuition
Have you ever wondered what is it that can make humans breathe, walk, make decisions, respond to some stimulus and environment, and ultimately think? It is quite clear now that the answer to this is our brain (as well as our spinal cord). Our central nervous system is composed of ... 
## Underlying Mathematics
### Forward Propagation
```math
\begin{aligned}
y_1 = w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + ... + w_{1i}x_i + b_1 \\
y_2 = w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + ... + w_{2i}x_i + b_2 \\
y_3 = w_{31}x_1 + w_{32}x_2 + w_{33}x_3 + ... + w_{3i}x_i + b_3 \\
\vdots \\
y_j = w_{j1}x_1 + w_{j2}x_2 + w_{j3}x_3 + ... + w_{ji}x_i + b_j \\
\end{aligned}
```
```math
\begin{aligned}
\begin{bmatrix} y_1 \\ y_2 \\ y_3 \\ \vdots \\ y_j \end{bmatrix} = \begin{bmatrix} w_{11} w_{12} w_{13} ... w_{1i} \\ w_{21} w_{22} w_{23} ... w_{2i} \\ w_{31} w_{32} w_{33} ... w_{3i} \\ \vdots \\ w_{j1} w_{j2} w_{j3} ... w_{ji} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_i \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ \vdots \\ b_j \end{bmatrix}
\end{aligned}
```
### Gradient Descent
### Backward Propagation

## Breaking into Modules

## License
This code is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing
Feel free to fork this repository and submit pull requests. Any contributions are welcome!

## Author
This repository was created by [Sorawit Chokphantavee](https://github.com/SorawitChok) and [Sirawit Chokphantavee](https://github.com/SirawitC).
