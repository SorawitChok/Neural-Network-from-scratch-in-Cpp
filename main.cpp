#include <stdio.h>
#include <vector>
#include "losses.cpp"

int main()
{
    vector<int> true_label = {0,1,0,0};
    vector<double> pred_prob = {0.3, 0.7, 0.2, 0.3};
    double loss = binaryCrossEntropy(true_label, pred_prob);
    printf("%lf",loss);
    return 0;
}