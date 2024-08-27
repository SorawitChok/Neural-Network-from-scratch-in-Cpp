#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>

std::vector<std::vector<double>> uniformWeightInitializer(int rows, int cols)
{
    /**
     * @brief Initializes a matrix with uniform random weights between -1.0 and 1.0
     * @param[in] rows The number of rows in the matrix
     * @param[in] cols The number of columns in the matrix
     * @return A matrix with uniform random weights between -1.0 and 1.0
     */
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
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
    /**
     * @brief Initializes a vector of biases with uniform random weights between -1.0 and 1.0
     * @param[in] size The size of the vector
     * @return A vector of biases with uniform random weights between -1.0 and 1.0
     */
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<double> bias(size);

    for (int i = 0; i < size; ++i)
    {
        bias[i] = dis(gen);
    }
    return bias;
}

double dotProduct(std::vector<double> &v1, std::vector<double> &v2)
{
    /**
     * @brief Computes the dot product of two vectors
     * @param[in] v1 The first vector
     * @param[in] v2 The second vector
     * @return The dot product of the two vectors
     */
    double result = 0;
    for (int i = 0; i < v1.size(); i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> &m)
{
    /**
     * @brief Computes the transpose of a matrix
     * @param[in] m The matrix to transpose
     * @return The transpose of the matrix
     */
    std::vector<std::vector<double>> trans_vec(m[0].size(), std::vector<double>());

    for (int i = 0; i < m.size(); i++)
    {
        for (int j = 0; j < m[i].size(); j++)
        {
            if (trans_vec[j].size() != m.size())
                trans_vec[j].resize(m.size());
            trans_vec[j][i] = m[i][j];
        }
    }
    return trans_vec;
}

std::vector<double> subtract(std::vector<double> &v1, std::vector<double> &v2)
{
    /**
     * @brief Computes the element-wise subtraction of two vectors
     * @param[in] v1 The first vector
     * @param[in] v2 The second vector
     * @return A new vector with the elementwise subtraction of v1 and v2
     */
    std::vector<double> out;
    std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(out), std::minus<double>());

    return out;
}

std::vector<double> scalarVectorMultiplication(std::vector<double> &v, double scalar)
{
    /**
     * @brief Computes the element-wise multiplication of a vector and a scalar
     * @param[in] v The vector to multiply
     * @param[in] scalar The scalar to multiply the vector with
     * @return A new vector with the element-wise multiplication of v and scalar
     */
    std::transform(v.begin(), v.end(), v.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar));
    return v;
}
