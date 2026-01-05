#include "problem.hpp"


Problem::Problem(
    const Problem &problem
) :
bound_type(problem.bound_type), 
costs(problem.costs), 
upper_bound(problem.upper_bound), 
lower_bound(problem.lower_bound), 
A(problem.A),
RHS(problem.RHS)
{
    checkConstraints();
    this -> problem_size = problem.problem_size;
    this -> constraints_size = problem.constraints_size;
    this -> logicals_size = problem.logicals_size;
}


Problem::Problem(
    const BoundaryTypeVector& _bound_type,
    const BoundaryTypeVector& _range_type,
    const ValuesVector& _costs,
    const ValuesVector& _lower_range,
    const ValuesVector& _upper_range,
    const ValuesVector& _lower_bound,
    const ValuesVector& _upper_bound,
    const Matrix& _A
) : 
bound_type(_bound_type), 
costs(_costs), 
upper_bound(_upper_bound), 
lower_bound(_lower_bound), 
A(_A)
{
    transformToComputeForm(_lower_range, _upper_range, _range_type);

    checkConstraints();
    
    size_t m = std::get<0>(this->A.getSize());
    size_t n = std::get<1>(this->A.getSize());

    this -> problem_size = n;
    this -> constraints_size = m;
    this -> logicals_size = n - m;
}


void Problem::transformToComputeForm(
    const ValuesVector &lower_range,
    const ValuesVector &upper_range,
    const BoundaryTypeVector &range_type
)
{
    size_t m = std::get<0>(A.getSize());
    size_t n = std::get<1>(A.getSize());
    A = A.stackColumns(linalg::ones(m));

    for (size_t i = 0; i < m; i++)
    {
        upper_bound.pushBack(-lower_range[i]);
        lower_bound.pushBack(-upper_range[i]);
        costs.pushBack(0);
        
        switch (range_type[i])
        {
        case BoundaryType::Fixed:
            bound_type.push_back(BoundaryType::Fixed);
            break;
     
        case BoundaryType::Boxed:
            bound_type.push_back(BoundaryType::Boxed);
            break;
        
        case BoundaryType::Lower:
            bound_type.push_back(BoundaryType::Upper);
            break;
        
        case BoundaryType::Upper:
            bound_type.push_back(BoundaryType::Lower);   
            break;
        }
    }  
    RHS = ValuesVector(m); 
}


void Problem::checkConstraints()
{
    size_t m = std::get<0>(A.getSize());
    size_t n = std::get<1>(A.getSize());
    size_t rank = A.rank();

    if (m != rank || m > n)
    {
        throw "Incorrect constraint size";
        exit(1);
    }
}


void Problem::setBoundary(const size_t& idx, const std::string& bound_type_name)
{
    auto boundary_type = stringToBoundaryType(bound_type_name);
    if (boundary_type == BoundaryType::UNKNOWN)
        throw "Unkown boundary type";
    bound_type[idx] = boundary_type; 
}


BoundaryType Problem::stringToBoundaryType(
    const std::string& bound_type_name
) const
{
    static const std::unordered_map<std::string, BoundaryType> bound_type_map = {
        {"fixed", BoundaryType::Fixed},
        {"free", BoundaryType::Free},
        {"boxed", BoundaryType::Boxed},
        {"upper", BoundaryType::Upper},
        {"lower", BoundaryType::Lower},
    };
    
    auto it = bound_type_map.find(bound_type_name);
    if (it != bound_type_map.end()) {
        return it->second;
    }
    return BoundaryType::UNKNOWN;
}


std::string Problem::boundaryTypeToString(
    const BoundaryType& bound_type_name
) const
{
    static const std::unordered_map<BoundaryType, std::string> bound_type_map = {
        {BoundaryType::Fixed, "fixed"},
        {BoundaryType::Free, "free"},
        { BoundaryType::Boxed, "boxed"},
        {BoundaryType::Upper, "upper"},
        {BoundaryType::Lower, "lower"},
    };
    
    auto it = bound_type_map.find(bound_type_name);
    if (it != bound_type_map.end()) {
        return it->second;
    }
    return "unknown";
}


void Problem::show()
{
    std::cout << "costs:" << std::endl;
    costs.show();
    std::cout << "constraint matrix:" << std::endl;
    A.show();
    std::cout << "rhs vector:" << std::endl;
    RHS.show();
    std::cout << "boundary:" << std::endl;
    for (size_t i = 0; i < problem_size; i++)
    {
        std::cout << boundaryTypeToString(bound_type[i]) << "  " << lower_bound[i] << "  " << upper_bound[i] <<std::endl;
    }
}