#include <vector>
#include "activation.cpp"
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