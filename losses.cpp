#include <vector>
#include <math.h>
#include <cmath>
#include "utils.cpp"

double BCELoss(std::vector<int> true_label, std::vector<double> pred_prob)
{
    double sum = 0;
    for (int i = 0; i < pred_prob.size(); i++)
    {
        sum += true_label[i] * log(pred_prob[i]) + (1 - true_label[i]) * log((1 - pred_prob[i]));
    }
    int size = true_label.size();
    double loss = -(1.0 / size) * sum;
    return loss;
}

std::vector<double> BCELossDerivative(std::vector<double> true_label, std::vector<double> pred_prob)
{
    std::vector<double> dev = {(pred_prob[0] - true_label[0])/((pred_prob[0])*(1-pred_prob[0]))};
    return dev;
}

double MSELoss(std::vector<double> true_label, std::vector<double> pred)
{
    double sum = 0;
    for (int i = 0; i < true_label.size(); i++)
    {
        sum += pow(true_label[i] - pred[i], 2.0);
    }
    int size = true_label.size();
    double loss = (1.0 / size) * sum;
    return loss;
}

std::vector<double> MSELossDerivative(std::vector<double> true_label, std::vector<double> pred)
{
    std::vector<double> sub = subtract(pred, true_label);
    std::vector<double> dev = scalarVectorMultiplication(sub, 2);
    return dev;
}
