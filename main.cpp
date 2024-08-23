#include <iostream>
#include <vector>
#include <functional> 
// #include "layer.cpp"
#include "utils.cpp"

int main()
{
    // Linear l1 = Linear(2,3);
    // Sigmoid a1 = Sigmoid();
    // Linear l2 = Linear(3,1);
    // Sigmoid a2 = Sigmoid();
    // Layer x[] = {l1,a1,l2,a2};

    std::vector<double> myv1 = {1,2,3.5};

    scalarVectorMultiplication(myv1, 3);

    printf("%lf %lf %lf", myv1[0], myv1[1], myv1[2]);
    // for (Layer i: x){
    //     i.forward();
    // }
}