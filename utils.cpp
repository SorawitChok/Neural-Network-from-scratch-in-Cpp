#include <cmath>
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