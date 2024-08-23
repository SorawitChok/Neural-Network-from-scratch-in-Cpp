#include <vector>
#include "activation.cpp"
#include "utils.cpp"

class Layer
{
public:
    std::vector<double> input;
    std::vector<double> output;
};

class Sigmoid : public Layer
{
public:
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

class Relu : public Layer
{
public:
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

class LeakyRelu : public Layer
{
public:
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

class Tanh : public Layer
{
public:
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

class Linear : public Layer
{
public:
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
            output.push_back(dotProduct(weights[i], input) + bias[i]);
        }

        return output;
    }
    // std::vector<double> backward(std::vector<double> error, double learning_rate)
    // {
    //     std::vector<double> input_error;  // dE/dX
    //     std::vector<double> weight_error; // dE/dW
    //     std::vector<double> bias_error;   // dE/dB

    //     std::vector<std::vector<double>> weight_transpose = transpose(weights);
    //     bias_error = error;

    //     for (int i = 0; i < weight_transpose.size(); i++)
    //     {
    //         input_error[i] = dotProduct(weight_transpose[i], error);
    //     }

    //     for (int j = 0; j < error.size(); j++)
    //     {
    //         for (int i = 0; i < input.size(); i++)
    //         {
    //             weight_error[j][i] = error[j] * input[i];
    //         }
    //     }

    //     bias = subtract(bias, learning_rate * bias_error);
    //     for (int i; weight_error.size(); i++)
    //     {
    //         weights[i] = subtract(weights[i], learning_rate * weight_error[i]);
    //     }
    //     return input_error;
    // }
};