#include <iostream>
#include <vector>
#include "layer.cpp"

int main()
{
    Layer *l[] = {new Linear(2, 3), new Relu(), new Linear(3, 1), new Sigmoid()};
    std::vector<double> output = {1.123, 2.223};
    for (int i = 0; i < sizeof(l) / sizeof(l[0]); i++)
    {
        output = l[i]->forward(output);
    }

    printf("%lf\n", output[0]);
}