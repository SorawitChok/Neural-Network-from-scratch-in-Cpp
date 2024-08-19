#include <vector>
#include "activation.cpp"

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
    std::vector<double> backward(std::vector<double> error)
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
    std::vector<double> backward(std::vector<double> error)
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
    std::vector<double> backward(std::vector<double> error)
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
    std::vector<double> backward(std::vector<double> error)
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