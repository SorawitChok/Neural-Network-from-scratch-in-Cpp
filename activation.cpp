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

vector<double> sigmoid(const vector<double> &x)
{
    vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(sigmoid(i));
    return result;
}

vector<double> sigmoidDerivative(const vector<double> &x)
{
    vector<double> result;
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

vector<double> relu(const vector<double> &x)
{
    vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(relu(i));
    return result;
}

vector<double> reluDerivative(const vector<double> &x)
{
    vector<double> result;
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

vector<double> leakyRelu(const vector<double> &x, double alpha = 0.01)
{
    vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(leakyRelu(i, alpha));
    return result;
}

vector<double> leakyReluDerivative(const vector<double> &x, double alpha = 0.01)
{
    vector<double> result;
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

vector<double> tanh(const vector<double> &x)
{
    vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(tanh(i));
    return result;
}

vector<double> tanhDerivative(const vector<double> &x)
{
    vector<double> result;
    result.reserve(x.size());
    for (double i : x)
        result.push_back(tanhDerivative(i));
    return result;
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