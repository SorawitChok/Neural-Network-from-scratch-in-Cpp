#include <vector>
#include <memory>
#include <iostream>
#include "layer.cpp"
#include "losses.cpp"

class NN
{
public:
    std::vector<std::unique_ptr<Layer>> layers;

    // Add layers dynamically
    void add(Layer *layer)
    {
        layers.emplace_back(layer);
    }

    // Make prediction using feed forward process
    std::vector<double> predict(std::vector<double> input)
    {
        return forward_propagation(input);
    }

    // Forward propagation
    std::vector<double> forward_propagation(const std::vector<double> input)
    {
        std::vector<double> output = input;
        for (const auto &layer : layers)
        {
            output = layer->forward(output);
        }
        return output;
    }

    // Backward propagation
    void backward_propagation(const std::vector<double> &error, double learning_rate)
    {
        std::vector<double> grad = error;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it)->backward(grad, learning_rate);
        }
    }

    // Training function
    void fit(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &y, int epochs, double learning_rate)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double total_loss = 0.0;
            for (size_t i = 0; i < X.size(); ++i)
            {
                // Forward pass
                std::vector<double> output = forward_propagation(X[i]);

                // Compute loss (assuming mean squared error)
                double loss = BCELoss(y[i], output);
                total_loss += loss;

                std::vector<double> loss_derivative = BCELossDerivative(y[i], output);
                // Backward pass
                backward_propagation(loss_derivative, learning_rate);
            }

            // Print loss for monitoring
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << total_loss / X.size() << std::endl;
        }
    }
};
