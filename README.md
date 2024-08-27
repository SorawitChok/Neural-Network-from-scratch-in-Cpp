# Neural Network from scratch in C++

Welcome to the **Neural Network from Scratch in C++** project! This repository features a straightforward implementation of a neural network built entirely from the ground up using C++. Designed to engage AI and machine learning enthusiasts, this project provides a hands-on opportunity to explore the mathematical and programming principles behind neural networks. Whether you're a learner or an experienced developer, you'll gain deeper insights into the inner workings of neural networks and their underlying algorithms.

## Table of Contents

- [Concept & Intuition](#Concept-&-Intuition)
- [Core Components](#Core-Components)
  - [Fully-connected layer](#Fully-connected-layer)
  - [Activation Function](#Activation-Function)
  - [Loss Function](#Loss-Function)
- [Underlying Mathematics](#Underlying-Mathematics)
  - [Forward Propagation](#Forward-Propagation)
  - [Gradient Descent](#Gradient-Descent)
  - [Backward Propagation](#Backward-Propagation)
- [Breaking into Modules](#Breaking-into-Modules)
- [Implementation](#Implementation)
- [License](#License)
- [Contributing](#Contributing)
- [Author](#Author)

## Concept & Intuition

Have you ever wondered what enables humans to breathe, walk, make decisions, respond to stimuli, and ultimately think? The answer lies in the brain and central nervous system, which consists of billions of interconnected neurons. Similarly, artificial neural networks (ANNs) are computational models inspired by the structure of the biological brain and neurons. They consist of interconnected layers of artificial neurons that process information and learn from data, enabling the network to make decisions and predictions.

In this project, we aim to build a neural network from scratch using C++, demystifying the concepts and mathematics behind these models. By manually implementing each component, we gain a deeper understanding of how neural networks operate and how they learn from data.

## Core Components

### Fully-connected layer

The fully-connected layer is arguably one of the most vital constituents of any neural network. The main functionality of this layer is to apply an affine transformation on the incoming data. But what exactly is this fancy term called affine transformation? Simply put, it is just a linear transformation (transforming a vector by multiplying it with a matrix) with a translation (adding another vector to a transformed vector). Mathematically, we can describe the output of this fully-connected layer (or affine transformation) as:

```math
y = Wx + b
```

Where:

- $W$ represents the weight matrix (perform a linear transformation on $x$),
- $x$ is the input vector,
- $b$ is the bias vector (translate $x$),
- $y$ is the output vector.

Although this transformation process is essential for the neural network, it is still very lacking in terms of its power, especially for processing highly complex data. The reason is that the affine transformation operations such as scaling, rotation, and shearing even with the translation still cannot account for the nonlinearity. Why is that?

let's see an example. Suppose we have a neural network with solely 2 fully-connected layers, then we can write out the equation as follows:

```math
\begin{aligned}
h_1 = W_1x + b_1----------------(1) \\
o = h_2 = W_2h_1 + b_2--------------(2) \\
\end{aligned}
```
Then if we substitute (1) into (2), this is what we get:
```math
\begin{aligned}
o = W_2(W_1x + b_1) + b_2 \\
o = W_2W_1x + W_2b_1 + b_2 \\
\end{aligned}
```
After that, we can group $W_2W_1$ into a new weight matrix $W'$ and $W_2b_1 + b_2$ into a new bias $b'$. Therefore, we end up with:

```math
o = w'x + b'
```
As you can see, it looks just like another affine transformation, which implies that no matter how many layers you put into your network, without a nonlinearity, the network will not be capable of exerting any more complex processing aside from a mere affine transformation (you can consult this [video](https://www.youtube.com/watch?v=JtVRC4qwmqg) for more explanation). This is why the activation function needs to come into play.


### Activation Functions

**Sigmoid Function**

```math
\sigma(x)= \frac {1}{1+e^{-x}}
```

**Rectified Linear Unit (ReLU)**

```math
ReLU(x) = max(0,x) = \begin{cases}
        x,&  \text{if } x > 0\\
        0,&   \text{otherwise}
\end{cases}
```

**Hyperbolic Tangent Function (Tanh)**

```math
tanh(x) = \frac{(e^x − e^{-x})}{(e^x + e^{-x})}
```

### Loss Functions

## Underlying Mathematics

### Forward Propagation

Forward propagation is the process by which input data is passed through the network to generate an output or prediction. The input values will be processed in each layer, and those processed values (output of the previous layer) will be passed as input for the subsequent layer. This process will continue until the data passes through all of the layers.

Let's see an example. Suppose we have the following neural network architecture:

<p align="center">
  <img src="./Images/Forward_propagation_math.png" alt="Forward"/>
</p>

Then, we can write out the forward propagation computation of this network as mathematical equations as follows:

```math
\begin{aligned}
h_1 = w_{h_1x_1}x_1 + w_{h_1x_2}x_2 + b_1 \\
h_2 = w_{h_2x_1}x_1 + w_{h_2x_2}x_2 + b_2 \\
h_3 = w_{h_3x_1}x_1 + w_{h_3x_2}x_2 + b_3 \\
\\
a_1 = \sigma(h_1) \\
a_2 = \sigma(h_2) \\
a_3 = \sigma(h_3) \\
\\
h_o = w_{oa_1}a_1 + w_{oa_2}a_2 + w_{oa_3}a_3 +  b_o \\
y = a_o = \sigma(h_o) \\
\end{aligned}
```

### Gradient Descent

### Backward Propagation

## Breaking into Modules

### Linear layer

Suppose, we have the following neural network linear(dense) layer taking in $i$ number of inputs and producing $j$ number of outputs.

<p align="center">
  <img src="./Images/NN_forward_ex.png" alt="NN"/>
</p>

Then for this particular layer, we can formalize it into a total of $j$ equations below.

```math
\begin{aligned}
y_1 = w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + ... + w_{1i}x_i + b_1 \\
y_2 = w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + ... + w_{2i}x_i + b_2 \\
y_3 = w_{31}x_1 + w_{32}x_2 + w_{33}x_3 + ... + w_{3i}x_i + b_3 \\
\vdots \\
y_j = w_{j1}x_1 + w_{j2}x_2 + w_{j3}x_3 + ... + w_{ji}x_i + b_j \\
\end{aligned}
```

In matrix form:

```math
\begin{aligned}
\begin{bmatrix} y_1 \\ y_2 \\ y_3 \\ \vdots \\ y_j \end{bmatrix} = \begin{bmatrix} w_{11} & w_{12} & w_{13} & ...  & w_{1i} \\ w_{21} & w_{22} & w_{23} & ... & w_{2i} \\ w_{31} & w_{32} & w_{33} & ... & w_{3i} \\ \vdots & \vdots & \vdots & \vdots & \vdots \\ w_{j1} & w_{j2} & w_{j3} & ... & w_{ji} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_i \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ \vdots \\ b_j \end{bmatrix} \\
\mathbf{y}_{j \times 1} = W_{j \times i} x_{i \times 1} + b_{j \times 1}
\end{aligned}
```

Where:

- $W_{j \times i}$ represents the weight matrix,
- $x_{i \times 1}$ is the input vector,
- $b_{j \times 1}$ is the bias vector,
- $\mathbf{y}_{j \times 1}$ is the output vector.

## Implementation

Now let's dive into the actual implementation of each module and function necessary for creating your own neural network.

### Utility Function

We will start with the important math operation and how we can implement it in C++. For these functions, you will need to import the following dependencies.

```cpp
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>
```

**Compute dot product**

This function allows you to compute the dot product between two input vectors, namely $\mathbf{v_1}$ and $\mathbf{v_2}$, and return a scalar number ($\mathbf{v_1} \cdot \mathbf{v_2}$) as an output.

```cpp
double dotProduct(std::vector<double> &v1, std::vector<double> &v2)
{
    /**
     * @brief Computes the dot product of two vectors
     * @param[in] v1 The first vector
     * @param[in] v2 The second vector
     * @return The dot product of the two vectors
     */
    double result = 0;
    for (int i = 0; i < v1.size(); i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}
```

**Element-wise multiplication between a vector and a scalar**

This function allows you to perform an element-wise multiplication between a vector $\mathbf{v}$ and one scalar number $a$, returning a modified vector $a\mathbf{v}$.

```cpp
std::vector<double> scalarVectorMultiplication(std::vector<double> &v, double scalar)
{
    /**
     * @brief Computes the element-wise multiplication of a vector and a scalar
     * @param[in] v The vector to multiply
     * @param[in] scalar The scalar to multiply the vector with
     * @return A new vector with the element-wise multiplication of v and scalar
     */
    std::transform(v.begin(), v.end(), v.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar));
    return v;
}
```

**Vector subtraction**

This function allows you to easily compute the subtraction between two vectors $\mathbf{v_1}$ and $\mathbf{v_2}$, resulting in a new vector with the value of $\mathbf{v_1} - \mathbf{v_2}$.

```cpp
std::vector<double> subtract(std::vector<double> &v1, std::vector<double> &v2)
{
    /**
     * @brief Computes the element-wise subtraction of two vectors
     * @param[in] v1 The first vector
     * @param[in] v2 The second vector
     * @return A new vector with the elementwise subtraction of v1 and v2
     */
    std::vector<double> out;
    std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(out), std::minus<double>());

    return out;
}
```

**Matrix transpose**

This function receives a matrix $\mathbf{M}$ as an input and returns the transpose of such matrix $\mathbf{m^T}$.

```cpp
std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> &m)
{
    /**
     * @brief Computes the transpose of a matrix
     * @param[in] m The matrix to transpose
     * @return The transpose of the matrix
     */
    std::vector<std::vector<double>> trans_vec(m[0].size(), std::vector<double>());

    for (int i = 0; i < m.size(); i++)
    {
        for (int j = 0; j < m[i].size(); j++)
        {
            if (trans_vec[j].size() != m.size())
                trans_vec[j].resize(m.size());
            trans_vec[j][i] = m[i][j];
        }
    }
    return trans_vec;
}
```

**Weights initialization**

This function allows you to generate a 2D vector of size $rows \times cols$ with a random value between -1.0 and 1.0. This function will be further use to generate the weights of the fully-connected layer (Linear layer).

```cpp
std::vector<std::vector<double>> uniformWeightInitializer(int rows, int cols)
{
    /**
     * @brief Initializes a matrix with uniform random weights between -1.0 and 1.0
     * @param[in] rows The number of rows in the matrix
     * @param[in] cols The number of columns in the matrix
     * @return A matrix with uniform random weights between -1.0 and 1.0
     */
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<std::vector<double>> weights(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            weights[i][j] = dis(gen);
        }
    }

    return weights;
}
```

**Bias initialization**

This function is used to generate a vector with a random value ranging from -1.0 to 1.0. This function will be further used to generate the bias of the fully connected layer (Linear layer).

```cpp
std::vector<double> biasInitailizer(int size)
{
    /**
     * @brief Initializes a vector of biases with uniform random weights between -1.0 and 1.0
     * @param[in] size The size of the vector
     * @return A vector of biases with uniform random weights between -1.0 and 1.0
     */
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<double> bias(size);

    for (int i = 0; i < size; ++i)
    {
        bias[i] = dis(gen);
    }
    return bias;
}
```

### Activation Function

Next, we will look into how we can implement each activation function that will allow our neural network perform a non-linear transformation on the input data. For more information and implementation of vectorize version of each activation function, you can consult [activation.cpp](./activation.cpp) file.

**Sigmoid and its derivative**

```cpp
double sigmoid(double x)
{
    /**
     * The sigmoid function maps any real-valued number to a value between 0 and 1.
     * It is often used in the output layer of a neural network when the task is a
     * binary classification problem.
     * @param x the input value
     * @return the output value of the sigmoid function
     */
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x)
{ /**
   * The derivative of the sigmoid function.
   * @param x the input value
   * @return the output value of the derivative of the sigmoid function
   */
    return exp(x) / pow((exp(x) + 1), 2);
}
```

**ReLU and its derivative**

```cpp
double relu(double x)
{ /**
   * The Rectified Linear Unit (ReLU) activation function.
   * @param x the input value
   * @return the output value of the ReLU function
   */
    if (x > 0)
        return x;
    else
        return 0;
}

double reluDerivative(double x)
{ /**
   * The derivative of the Rectified Linear Unit (ReLU) activation function.
   * @param x the input value
   * @return the output value of the derivative of the ReLU function
   */
    if (x >= 0)
        return 1;
    else
        return 0;
}
```

**Leaky ReLU and its derivative**

```cpp
double leakyRelu(double x, double alpha = 0.01)
{
    /**
     * The Leaky Rectified Linear Unit (Leaky ReLU) activation function.
     * @param x the input value
     * @param alpha the leak rate, defaults to 0.01
     * @return the output value of the Leaky ReLU function
     */
    if (x > 0)
        return x;
    else
        return alpha * x;
}

double leakyReluDerivative(double x, double alpha = 0.01)
{ /**
   * The derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
   * @param x the input value
   * @param alpha the leak rate, defaults to 0.01
   * @return the output value of the derivative of the Leaky ReLU function
   */
    if (x >= 0)
        return 1;
    else
        return alpha;
}
```

**Tanh and its derivative**

```cpp
double tanh(double x)
{ /**
   * The Hyperbolic Tangent (tanh) activation function.
   * @param x the input value
   * @return the output value of the tanh function
   */
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double tanhDerivative(double x)
{ /**
   * The derivative of the Hyperbolic Tangent (tanh) activation function.
   * @param x the input value
   * @return the output value of the derivative of the tanh function
   */
    return 1 - pow(tanh(x), 2);
}
```

### Layers

### Loss Function

## License

This code is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Feel free to fork this repository and submit pull requests. Any contributions are welcome!

## Author

This repository was created by [Sorawit Chokphantavee](https://github.com/SorawitChok) and [Sirawit Chokphantavee](https://github.com/SirawitC).
