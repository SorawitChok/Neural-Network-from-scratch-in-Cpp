#include <vector>
#include <math.h>
using namespace std;

double binaryCrossEntropy(vector<int> true_label, vector<double> pred_prob){
    double sum = 0;
    for(int i=0; i < pred_prob.size() ; i++){
        sum += true_label[i]*log10(pred_prob[i]) + (1-true_label[i])*log10((1-pred_prob[i]));
    }
    int size = true_label.size();
    double loss = -(1.0/size)*sum; 
    return loss;
}