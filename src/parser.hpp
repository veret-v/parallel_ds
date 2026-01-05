#pragma once

#include <vector>
#include <string>
#include <cmath>

#include <CoinMpsIO.hpp>

#include "problem.hpp"
#include "types.hpp"

#define bound_inf 1e+100

class LPparser
{
private:
    BoundaryType mpsTypeToBoundaryType(const char bound_type_name) const;
    inline bool isinf_bound(const double x) const {return (x > bound_inf || x < -bound_inf) ? true : false;};
   
public:
    LPparser() {};
    ~LPparser() {};

    int readMps(const std::string& file_name, Problem& problem);
};

