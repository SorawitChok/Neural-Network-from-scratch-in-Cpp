#include <iostream>
#include <vector>
#include "activation.cpp"
#include "losses.cpp"

using namespace std;
int main()
{
    vector<double> true_lab = {3, -0.5, 2, 7};
    vector<double> pred_lab = {2.5, 0.0, 2, 8};
    double loss = MSELoss(true_lab, pred_lab);
    printf("%lf",loss);
}