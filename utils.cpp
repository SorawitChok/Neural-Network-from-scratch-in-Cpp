#include <vector>
#include <random>

std::vector<std::vector<double>> uniformWeightInitializer(int rows, int cols)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<std::vector<double>> weights(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            weights[i][j] = dis(gen);
        }
    }

    return weights;
}

std::vector<double> biasInitailizer(int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<double> bias(size);

    for (int i = 0; i < size; ++i)
    {
        bias[i] = dis(gen);
    }
    return bias;
}

double dotProduct(std::vector<double> v1, std::vector<double> v2)
{
    double result = 0;
    for (int i = 0; i<v1.size(); i++)
    {
        result += v1[i]*v2[i];
    }
    return result;
}