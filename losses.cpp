#include <vector>
#include <math.h>
#include <cmath>

double BCELoss(std::vector<double> true_label, std::vector<double> pred_prob)
{ /**
   * Binary Cross Entropy Loss
   * @param true_label true labels of the data
   * @param pred_prob predicted probabilities
   * @return binary cross entropy loss
   */
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
{ /**
   * Compute derivative of binary cross entropy loss
   * @param true_label true labels of the data
   * @param pred_prob predicted probabilities
   * @return derivative of binary cross entropy loss
   */
    std::vector<double> dev = {(pred_prob[0] - true_label[0]) / ((pred_prob[0]) * (1 - pred_prob[0]))};
    return dev;
}

double MSELoss(std::vector<double> true_label, std::vector<double> pred)
{ /**
   * Mean Squared Error Loss
   * @param true_label true labels of the data
   * @param pred predicted values
   * @return mean squared error loss
   */
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
{ /**
   * Compute derivative of mean squared error loss
   * @param true_label true labels of the data
   * @param pred predicted values
   * @return derivative of mean squared error loss
   */
    std::vector<double> sub = subtract(pred, true_label);
    std::vector<double> dev = scalarVectorMultiplication(sub, 2);
    return dev;
}
