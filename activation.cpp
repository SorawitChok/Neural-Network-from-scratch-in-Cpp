#include <cmath>
#include <vector>
using namespace std;
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double relu(double x)
{
    if (x > 0)
        return x;
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

vector<double> softmax(vector<double> z)
{
    vector<double> result;
    double sum = 0.0;
    for (double i : z)
        sum += exp(i);
    for (int j = 0; j < z.size(); j++)
        result.push_back(exp(z[j]) / sum);
    return result;
}