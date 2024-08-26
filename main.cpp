#include <iostream>
#include <vector>
#include "NN.cpp"

int main()
{
    NN neural_network;

    // Add layers dynamically
    neural_network.add(new Linear(2, 3));
    neural_network.add(new Relu());
    neural_network.add(new Linear(3, 1));
    neural_network.add(new Sigmoid());

    // Example input data
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};

    // Train the network
    neural_network.fit(X, y, 10, 0.01);

    return 0;
}
