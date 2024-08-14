#include <vector>
#include <math.h>
#include<cmath>
using namespace std;

double BCELoss(vector<int> true_label, vector<double> pred_prob){
    double sum = 0;
    for(int i=0; i < pred_prob.size() ; i++){
        sum += true_label[i]*log10(pred_prob[i]) + (1-true_label[i])*log10((1-pred_prob[i]));
    }
    int size = true_label.size();
    double loss = -(1.0/size)*sum; 
    return loss;
}

double MSELoss(vector<double> true_label, vector<double> pred){
    double sum = 0;
    for(int i=0; i < true_label.size() ; i++){
        sum += pow(true_label[i] - pred[i],2.0);
    }
    int size = true_label.size();
    double loss = (1.0/size)*sum;
    return loss;
}
