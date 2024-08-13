#include <cmath>
#include <vector>

using namespace std;
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x)
{
    return exp(x) / pow((exp(x) + 1), 2);
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

double tanh(double x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double tanhDerivative(double x)
{
    return 1 - pow(tanh(x), 2);
}

// vector<double> softmax(vector<double> z)
// {
//     vector<double> result;
//     double sum = 0.0;
//     for (double i : z)
//         sum += exp(i);
//     for (int j = 0; j < z.size(); j++)
//         result.push_back(exp(z[j]) / sum);
//     return result;
// }