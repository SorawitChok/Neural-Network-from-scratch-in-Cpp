#include <iostream>
#include <vector>
// #include "layer.cpp"
// #include "losses.cpp"
#include "utils.cpp"

int main()
{
    std::vector<std::vector<double>> test = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<std::vector<double>> x = transpose(test);

    printf("%lf %lf\n", test[0][0], test[0][1]);
    printf("%lf %lf\n", test[1][0], test[1][1]);
    printf("%lf %lf\n", test[2][0], test[2][1]);
    printf("======================\n");
    printf("%lf %lf %lf\n", x[0][0], x[0][1], x[0][2]);
    printf("%lf %lf %lf\n", x[1][0], x[1][1], x[1][2]);
}