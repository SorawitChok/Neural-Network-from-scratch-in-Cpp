#include "activation.cpp"
class Relu
{
public:
    double forward(double x)
    {
        if (x > 0)
            return x;
        else
            return 0;
    }
    double backward(double x)
    {
        if (x > 0)
            return 1;
        else
            return 0;
    }
}