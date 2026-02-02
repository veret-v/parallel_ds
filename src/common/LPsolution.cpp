#include "LPsolution.hpp"



void LPsolution::show()
{
    std::cout << "-----------------------------LP result-----------------------------" << std::endl;
    std::cout << "solved:                     " << solved << std::endl;
    std::cout << "status:                     " << message << std::endl;
    std::cout << "Z(obj. func. value):        " << Z << std::endl;
    std::cout << "phase 2 time:               " << phase2_time << std::endl;
    std::cout << "iterations:                 " << iterations << std::endl;
    std::cout << "x0:                         " << x[0] << std::endl;
    for (size_t i = 1; i < std::min(int(x.getSize()), 10); i++)
        std::cout << "                            " <<  x[i] << std::endl;
    if (x.getSize() > 10)
        std::cout << "                            " <<  "..." << std::endl;
    
}


