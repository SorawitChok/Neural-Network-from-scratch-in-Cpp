#include <vector>
#include "activation.cpp"
#include "utils.cpp"

// Define Layer based class
class Layer
{
public:
    std::vector<double> input;
    std::vector<double> output;
    virtual std::vector<double> forward(const std::vector<double> input_data) = 0;
    virtual std::vector<double> backward(std::vector<double> error, double learning_rate) = 0;
};

// Define Sigmoid class inherited from Layer
class Sigmoid : public Layer
{
public:
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vectSigmoid(input);
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> derivative = vectSigmoidDerivative(input);
        std::vector<double> grad_input;
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

// Define Relu class inherited from Layer
class Relu : public Layer
{
public:
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vectRelu(input);
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> derivative = vectReluDerivative(input);
        std::vector<double> grad_input;
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

// Define LeakyRelu class inherited from Layer
class LeakyRelu : public Layer
{
public:
    double alpha = 0.01;
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vectLeakyRelu(input, alpha);
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> derivative = vectLeakyReluDerivative(input, alpha);
        std::vector<double> grad_input;
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

// Define Tanh class inherited from Layer
class Tanh : public Layer
{
public:
    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output = vectTanh(input);
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> derivative = vectTanhDerivative(input);
        std::vector<double> grad_input;
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

// Define Linear class (fully connected layer) inherited from Layer
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

    std::vector<double> forward(const std::vector<double> input_data) override
    {
        input = input_data;
        output.clear();
        for (int i = 0; i < output_neuron; i++)
        {
            output.push_back(dotProduct(weights[i], input) + bias[i]);
        }
        return output;
    }
    std::vector<double> backward(std::vector<double> error, double learning_rate) override
    {
        std::vector<double> input_error;               // dE/dX
        std::vector<std::vector<double>> weight_error; // dE/dW
        std::vector<double> bias_error;                // dE/dB
        std::vector<std::vector<double>> weight_transpose;
        weight_error.clear();
        bias_error.clear();
        input_error.clear();
        weight_transpose.clear();

        weight_transpose = transpose(weights);
        bias_error = error;
        for (int i = 0; i < weight_transpose.size(); i++)
        {
            input_error.push_back(dotProduct(weight_transpose[i], error));
        }
        for (int j = 0; j < error.size(); j++)
        {
            std::vector<double> row;
            for (int i = 0; i < input.size(); i++)
            {
                row.push_back(error[j] * input[i]);
            }
            weight_error.push_back(row);
        }

        std::vector<double> delta_bias = scalarVectorMultiplication(bias_error, learning_rate);
        bias = subtract(bias, delta_bias);
        for (int i = 0; i < weight_error.size(); i++)
        {
            std::vector<double> delta_weight = scalarVectorMultiplication(weight_error[i], learning_rate);
            weights[i] = subtract(weights[i], delta_weight);
        }

        return input_error;
    }
};