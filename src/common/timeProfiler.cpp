#include "timeProfiler.hpp"


TimeProfiler::TimeProfiler() :
    _init_time(0),
    _factor_time(0),
    _btran_time(0),
    _pivot_row_time(0),
    _ratio_test1_time(0),
    _ratio_test2_time(0),
    _update_red_cost_time(0),
    _ftran_time(0),
    _ftran_bfrt_time(0),
    _ftran_beta_time(0),
    _basis_update_time(0),
    _restore_proc_time(0)
{
    _prev_time = std::chrono::high_resolution_clock::now();
}


void TimeProfiler::startTimer() 
{
    _prev_time = std::chrono::high_resolution_clock::now();
}


void TimeProfiler::stopTimer(const ALgorithmPart& alg_part) 
{
    _curr_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(_curr_time - _prev_time).count();

    switch (alg_part)
    {
    case ALgorithmPart::Init:
        _init_time += duration_ms;
        break;
        
    case ALgorithmPart::Factor:
        _factor_time += duration_ms;
        break;

    case ALgorithmPart::Btran:
        _btran_time += duration_ms;
        break;

    case ALgorithmPart::PivotRow:
        _pivot_row_time += duration_ms;
        break;

    case ALgorithmPart::RatioTestPart1:
        _ratio_test1_time += duration_ms;
        break;

    case ALgorithmPart::Ftran:
        _ftran_time += duration_ms;
        break;

    case ALgorithmPart::FtranBfrt:
        _ftran_bfrt_time += duration_ms;
        break;

    case ALgorithmPart::FtranBeta:
        _ftran_beta_time += duration_ms;
        break;

    case ALgorithmPart::BasisUpate:
        _basis_update_time += duration_ms;
        break;

    case ALgorithmPart::RatioTestPart2:
        _ratio_test2_time += duration_ms;
        break;

    case ALgorithmPart::UpdateRedCosts:
        _update_red_cost_time += duration_ms;
        break;

    case ALgorithmPart::RestoreProc:
        _restore_proc_time += duration_ms;
        break;

    case ALgorithmPart::Pricing:
        _pricing_time += duration_ms;
        break;
    }
}


void TimeProfiler::reset()
{
    _init_time = 0.0;
    _factor_time = 0.0;
    _btran_time = 0.0;
    _pivot_row_time = 0.0;
    _ratio_test1_time = 0.0;
    _ratio_test2_time = 0.0;
    _update_red_cost_time = 0.0;
    _ftran_time = 0.0;
    _ftran_bfrt_time = 0.0;
    _ftran_beta_time = 0.0;
    _basis_update_time = 0.0;
    _restore_proc_time = 0.0;
    _pricing_time = 0.0;
}


void TimeProfiler::printInfo()
{
    _basis_update_time -= _ftran_beta_time; // basis update include beta ftran

    double total = _init_time + _factor_time + _btran_time + _pivot_row_time +
                   _ratio_test1_time + _ratio_test2_time + _update_red_cost_time +
                   _ftran_time + _ftran_bfrt_time + _ftran_beta_time +
                   _basis_update_time + _restore_proc_time + _pricing_time;

    std::cout << "\n========== Algorithm Time Profile ==========\n";
    std::cout << std::left << std::setw(20) << "Stage"
              << std::right << std::setw(12) << "Time (ns)"
              << std::setw(12) << "Percent"
              << "\n--------------------------------------------\n";

    auto printRow = [&](const std::string& name, double time) {
        double percent = (total > 0.0) ? (time / total * 100.0) : 0.0;
        std::cout << std::left << std::setw(20) << name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << time / 1e6
                  << std::setw(11) << std::setprecision(2) << percent << "%\n";
    };

    printRow("Initialization", _init_time);
    printRow("Factorization", _factor_time);
    printRow("Pricing", _pricing_time);
    printRow("Btran", _btran_time);
    printRow("Pivot Row", _pivot_row_time);
    printRow("Ratio Test (Part1)", _ratio_test1_time);
    printRow("Ftran", _ftran_time);
    printRow("Ftran (BFRT)", _ftran_bfrt_time);
    printRow("Ftran (Beta)", _ftran_beta_time);
    printRow("Basis Update", _basis_update_time);
    printRow("Ratio Test (Part2)", _ratio_test2_time);
    printRow("Update Reduced Costs", _update_red_cost_time);
    printRow("Restore Procedure", _restore_proc_time);

    std::cout << "--------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "TOTAL"
              << std::right << std::setw(12) << std::fixed << std::setprecision(3) << total / 1e6
              << std::setw(11) << "100.00%\n";
    std::cout << "============================================\n";
}