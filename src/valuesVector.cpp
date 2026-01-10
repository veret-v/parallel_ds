#include "valuesVector.hpp"


ValuesVector ValuesVector::operator-(const ValuesVector &values_vector) const
{
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    
    ValuesVector new_vec(values_vector.getSize());
    for (size_t i = 0; i < getSize(); i++)
       new_vec[i] = operator[](i) - values_vector[i];

    return new_vec;
}


ValuesVector ValuesVector::operator-() const
{
    ValuesVector new_vec(getSize());
    for (size_t i = 0; i < getSize(); i++)
       new_vec[i] =  -operator[](i);

    return new_vec;
}


double ValuesVector::dot(const ValuesVector &values_vector) const
{
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    double dot_val = 0;
    for (size_t i = 0; i < getSize(); i++)
        dot_val += operator[](i) * values_vector[i];

    return dot_val;
}


double ValuesVector::mean() const
{
    double sum = 0;
    for (auto a_i : data)
        sum +=  a_i;
    return sum / getSize();
}


double ValuesVector::norm() const
{
    double sum = 0;
    for (auto a_i : data)
        sum += pow(a_i, 2);
    return sum;
}


size_t ValuesVector::countNonZero() const
{
    double count = 0;
    for (auto a_i : data)
        count += (a_i == 0) ? 0 : 1;
    return count;
}

ValuesVector& ValuesVector::operator=(const ValuesVector& values_vector)
{
    data = values_vector.data;
    return *this;
}


ValuesVector& ValuesVector::operator-=(const ValuesVector& values_vector)
{
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    
    ValuesVector new_vec(values_vector.getSize());
    for (size_t i = 0; i < getSize(); i++)
        operator[](i) -= values_vector[i];

    return *this;   
}


ValuesVector& ValuesVector::operator+=(const ValuesVector& values_vector)
{
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    
    ValuesVector new_vec(values_vector.getSize());
    for (size_t i = 0; i < getSize(); i++)
        operator[](i) += values_vector[i];

    return *this;   
}


ValuesVector ValuesVector::operator+(const ValuesVector& values_vector) const
{
    if (values_vector.getSize() != getSize())
    {
        std::cerr << "Incorrect size" << std::endl;
        exit(1);
    }
    
    ValuesVector new_vec(values_vector.getSize());
    for (size_t i = 0; i < getSize(); i++)
       new_vec[i] = operator[](i) + values_vector[i];

    return new_vec;
}


ValuesVector ValuesVector::operator*(const double& value) const
{
    ValuesVector new_vec(getSize());
    for (size_t i = 0; i < getSize(); i++)
       new_vec[i] = value * operator[](i);

    return new_vec;
}


ValuesVector::ValuesVector(const size_t size)
{
    data = std::vector<double>(size);
}


ValuesVector::ValuesVector(const std::vector<double> vector)
{
    data = vector;
}


ValuesVector::ValuesVector(const ValuesVector& values_vector)
{
    data = values_vector.data;
}


void ValuesVector::pushBack(const double& value)
{
    data.push_back(value);
}


ValuesVector ValuesVector::operator()(const IndexVector& indexes) const
{
    ValuesVector new_vector(indexes.size());
    
    for (size_t i = 0; i < indexes.size(); i++)
        new_vector[i] = operator[](indexes[i]);
    
    return new_vector;
}


void ValuesVector::show() const
{
    std::cout << "Vector(" << getSize() << "):" << std::endl;
    for (size_t i = 0; i < getSize(); i++)
        std::cout << operator[](i) << " ";
    std::cout << std::endl;
}


void ValuesVector::setValues(const ValuesVector& values_vector, const IndexVector& index_vector)
{
    for (size_t i = 0; i < index_vector.size(); i++)
    {
        size_t j = index_vector[i];
        operator[](j) = values_vector[i];
    } 
}


ValuesVector ValuesVector::operator()(const size_t start, const size_t stop) const
{
    ValuesVector result(stop - start);
    for (size_t i = start; i < stop; i++)
        result[i] = operator[](i);
    return result;
}


ValuesVector::iterator ValuesVector::begin()
{
    return data.begin();
}

ValuesVector::iterator ValuesVector::end()
{
    return data.end();
}

ValuesVector::const_iterator ValuesVector::begin() const
{
    return data.begin();
}

ValuesVector::const_iterator ValuesVector::end() const
{
    return data.end();
}


