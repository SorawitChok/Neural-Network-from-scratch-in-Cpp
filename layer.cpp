#include <vector>
#include "activation.cpp"
#include "utils.cpp"

class Sigmoid
{
public:
    std::vector<double> input;
    std::vector<double> output;
    std::vector<double> forward(const std::vector<double> &input_data)
    {
        input = input_data;
        output = vectSigmoid(input);
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate)
    {
        std::vector<double> derivative = vectSigmoidDerivative(input);
        std::vector<double> grad_input;
        grad_input.reserve(derivative.size());
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

class Relu
{
public:
    std::vector<double> input;
    std::vector<double> output;
    std::vector<double> forward(const std::vector<double> &input_data)
    {
        input = input_data;
        output = vectRelu(input);
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate)
    {
        std::vector<double> derivative = vectReluDerivative(input);
        std::vector<double> grad_input;
        grad_input.reserve(derivative.size());
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

class LeakyRelu
{
public:
    std::vector<double> input;
    std::vector<double> output;
    double alpha = 0.01;
    std::vector<double> forward(const std::vector<double> &input_data)
    {
        input = input_data;
        output = vectLeakyRelu(input, alpha);
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate)
    {
        std::vector<double> derivative = vectLeakyReluDerivative(input, alpha);
        std::vector<double> grad_input;
        grad_input.reserve(derivative.size());
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

class Tanh
{
public:
    std::vector<double> input;
    std::vector<double> output;
    std::vector<double> forward(const std::vector<double> &input_data)
    {
        input = input_data;
        output = vectTanh(input);
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate)
    {
        std::vector<double> derivative = vectTanhDerivative(input);
        std::vector<double> grad_input;
        grad_input.reserve(derivative.size());
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

class Linear
{
public:
    std::vector<double> input;
    std::vector<double> output;
    int input_neuron;
    int output_neuron;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    Linear(int num_in, int num_out)
    {
        input_neuron = num_in;
        output_neuron = num_out;
        weights = uniformWeightInitializer(num_out, num_in);
        bias = biasInitailizer(num_out);
    }

    std::vector<double> forward(const std::vector<double> &input_data)
    {
        input = input_data;
        for (int i = 0; i < output_neuron; i++)
        {
            output[i] = dotProduct(weights[0], input);
        }

        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate)
    {
        std::vector<double> derivative = vectTanhDerivative(input);
        std::vector<double> grad_input;
        grad_input.reserve(derivative.size());
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};