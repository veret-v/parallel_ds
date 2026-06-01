#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

#include "types.hpp"


class TimeProfiler
{
private:
    using Time = std::chrono::high_resolution_clock::time_point;

    double _init_time;
    double _factor_time;
    double _btran_time;
    double _pivot_row_time;
    double _ratio_test1_time;
    double _ratio_test2_time;
    double _ftran_time;
    double _ftran_bfrt_time;
    double _ftran_beta_time;
    double _basis_update_time;
    double _update_red_cost_time;
    double _restore_proc_time;
    double _pricing_time;

    Time _prev_time;
    Time _curr_time;


public:
    TimeProfiler();

    void startTimer(); 
    void stopTimer(const ALgorithmPart& alg_part); 

    void printInfo();
    void reset();
};
