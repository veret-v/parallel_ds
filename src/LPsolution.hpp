#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <chrono>
#include <set>
#include <queue>
#include <algorithm>

#include "problem.hpp"
#include "linalg.hpp"
#include "valuesVector.hpp"
#include "matrix.hpp"
#include "types.hpp"


class LPsolution
{
public:
    double Z;
    bool solved;
    ValuesVector x;
    std::string message;
    size_t phase2_time;
    size_t iterations;

    LPsolution(
        const double _Z, 
        const ValuesVector& _x, 
        const bool _solved, 
        const std::string& _message
    ) : Z(_Z), x(_x), solved(_solved), message(_message) {};
    LPsolution(
        const LPsolution& solution
    ) : LPsolution(solution.Z, solution.x, solution.solved, solution.message) {};
    LPsolution() : LPsolution(0, ValuesVector(), false, "") {};

    void show();
};

