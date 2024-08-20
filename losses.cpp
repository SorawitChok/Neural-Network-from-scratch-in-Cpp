#include <vector>
#include <math.h>
#include <cmath>

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

double BCELossDerivative(int true_label, double pred_prob)
{
    double dev = (pred_prob - true_label)/((pred_prob)*(1-pred_prob));
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

double MSELossDerivative(double true_label, double pred)
{
    double dev = 2*(true_label-pred)*-1;
    return dev;
}
