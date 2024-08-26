#include <cmath>
#include <vector>

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x)
{
    return exp(x) / pow((exp(x) + 1), 2);
}

std::vector<double> vectSigmoid(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoid(i));
    return result;
}

std::vector<double> vectSigmoidDerivative(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoidDerivative(i));
    return result;
}

double relu(double x)
{
    if (x > 0)
        return x;
    else
        return 0;
}

double reluDerivative(double x)
{
    if (x >= 0)
        return 1;
    else
        return 0;
}

std::vector<double> vectRelu(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(relu(i));
    return result;
}

std::vector<double> vectReluDerivative(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(reluDerivative(i));
    return result;
}

double leakyRelu(double x, double alpha = 0.01)
{
    if (x > 0)
        return x;
    else
        return alpha * x;
}

double leakyReluDerivative(double x, double alpha = 0.01)
{
    if (x >= 0)
        return 1;
    else
        return alpha;
}

std::vector<double> vectLeakyRelu(const std::vector<double> x, double alpha = 0.01)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(leakyRelu(i, alpha));
    return result;
}

std::vector<double> vectLeakyReluDerivative(const std::vector<double> x, double alpha = 0.01)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(leakyReluDerivative(i, alpha));
    return result;
}

double tanh(double x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double tanhDerivative(double x)
{
    return 1 - pow(tanh(x), 2);
}

std::vector<double> vectTanh(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(tanh(i));
    return result;
}

std::vector<double> vectTanhDerivative(const std::vector<double> x)
{
    std::vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(tanhDerivative(i));
    return result;
}

// std::vector<double> softmax(std::vector<double> z)
// {
//     std::vector<double> result;
//     double sum = 0.0;
//     for (double i : z)
//         sum += exp(i);
//     for (int j = 0; j < z.size(); j++)
//         result.push_back(exp(z[j]) / sum);
//     return result;
// }