#include "problem.hpp"


template <typename MatrixType, typename VectorType>
Problem<MatrixType, VectorType>::Problem(
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


template <typename MatrixType, typename VectorType>
Problem<MatrixType, VectorType>::Problem(
    const BoundaryTypeVector& _bound_type,
    const BoundaryTypeVector& _range_type,
    const VectorType& _costs,
    const VectorType& _lower_range,
    const VectorType& _upper_range,
    const VectorType& _lower_bound,
    const VectorType& _upper_bound,
    const MatrixType& _A
) : 
bound_type(_bound_type), 
costs(_costs), 
upper_bound(_upper_bound), 
lower_bound(_lower_bound), 
A(_A)
{
    transformToComputeForm(_lower_range, _upper_range, _range_type);

    checkConstraints();
    
    int m = std::get<0>(this->A.getSize());
    int n = std::get<1>(this->A.getSize());

    this -> problem_size = n;
    this -> constraints_size = m;
    this -> logicals_size = n - m;
}


template <typename MatrixType, typename VectorType>
void Problem<MatrixType, VectorType>::transformToComputeForm(
    const VectorType &lower_range,
    const VectorType &upper_range,
    const BoundaryTypeVector &range_type
)
{
    int m = std::get<0>(A.getSize());
    int n = std::get<1>(A.getSize());
    A.stackColUnitMatrix();

    for (int i = 0; i < m; i++)
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
    RHS = VectorType(m); 
}


template <typename MatrixType, typename VectorType>
void Problem<MatrixType, VectorType>::checkConstraints()
{
    int m = std::get<0>(A.getSize());
    int n = std::get<1>(A.getSize());

    if (m > n)
    {
        throw "Incorrect constraint size";
        exit(1);
    }
}


template <typename MatrixType, typename VectorType>
void Problem<MatrixType, VectorType>::setBoundary(const int& idx, const std::string& bound_type_name)
{
    auto boundary_type = stringToBoundaryType(bound_type_name);
    if (boundary_type == BoundaryType::UNKNOWN)
        throw "Unkown boundary type";
    bound_type[idx] = boundary_type; 
}


template <typename MatrixType, typename VectorType>
BoundaryType Problem<MatrixType, VectorType>::stringToBoundaryType(
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


template <typename MatrixType, typename VectorType>
std::string Problem<MatrixType, VectorType>::boundaryTypeToString(
    const BoundaryType& bound_type_name
) const
{
    static const std::unordered_map<BoundaryType, std::string> bound_type_map = {
        {BoundaryType::Fixed, "fixed"},
        {BoundaryType::Free, "free"},
        {BoundaryType::Boxed, "boxed"},
        {BoundaryType::Upper, "upper"},
        {BoundaryType::Lower, "lower"},
    };
    
    auto it = bound_type_map.find(bound_type_name);
    if (it != bound_type_map.end()) {
        return it->second;
    }
    return "unknown";
}


template <typename MatrixType, typename VectorType>
void Problem<MatrixType, VectorType>::show()
{
    std::cout << "costs:" << std::endl;
    costs.show();
    std::cout << "constraint matrix:" << std::endl;
    A.show();
    std::cout << "rhs vector:" << std::endl;
    RHS.show();
    std::cout << "boundary:" << std::endl;
    for (int i = 0; i < problem_size; i++)
    {
        std::cout << boundaryTypeToString(bound_type[i]) << "  " << lower_bound[i] << "  " << upper_bound[i] <<std::endl;
    }
}


template <typename MatrixType, typename VectorType>
Problem<MatrixType, VectorType>::Problem(const std::string& filename)
{
    CoinMpsIO mps_reader;
    int status = mps_reader.readMps(filename.c_str());
    if (status == 0 || status == 1)
    {
        CoinPackedMatrix coeffs = *mps_reader.getMatrixByCol();
        int rows_num = coeffs.getNumRows();
        int cols_num = coeffs.getNumCols();

        A = coeffs;
       
        const double* lower_range_parsed  = mps_reader.getRowLower();
        const double* upper_range_parsed  = mps_reader.getRowUpper();
        const char*   types_parsed  = mps_reader.getRowSense();

        BoundaryTypeVector range_types(rows_num);
       
        VectorType range_lower(rows_num);
        VectorType range_upper(rows_num);

        for (int i = 0; i < rows_num; i++)
        {
            range_types[i] = mpsTypeToBoundaryType(types_parsed[i]);
            range_lower[i] = lower_range_parsed[i];
            range_upper[i] = upper_range_parsed[i];
        }

        const double* lower_bound_parsed  = mps_reader.getColLower();
        const double* upper_bound_parsed  = mps_reader.getColUpper();
        const double* costs_parsed        = mps_reader.getObjCoefficients();

        lower_bound = VectorType(cols_num);
        upper_bound = VectorType(cols_num);
        costs =       VectorType(cols_num);
        bound_type =  BoundaryTypeVector(cols_num);

        for (int i = 0; i < cols_num; i++)
        {
            lower_bound[i] = lower_bound_parsed[i];
            upper_bound[i] = upper_bound_parsed[i];
            costs[i]       = costs_parsed[i];

            if (isinf_bound(lower_bound[i]) && isinf_bound(upper_bound[i]))
                bound_type[i] = BoundaryType::Free;
            else if (isinf_bound(lower_bound[i]) && !isinf_bound(upper_bound[i]))
                bound_type[i] = BoundaryType::Upper;
            else if (!isinf_bound(lower_bound[i]) && isinf_bound(upper_bound[i]))
                bound_type[i] = BoundaryType::Lower;
            else if (!isinf_bound(lower_bound[i]) && !isinf_bound(upper_bound[i]) && lower_bound[i] == upper_bound[i])
                bound_type[i] = BoundaryType::Fixed;
            else
                bound_type[i] = BoundaryType::Boxed;
        }
        
        transformToComputeForm(range_lower, range_upper, range_types);
        checkConstraints();
        
        int m = std::get<0>(this->A.getSize());
        int n = std::get<1>(this->A.getSize());

        this -> problem_size = n;
        this -> constraints_size = m;
        this -> logicals_size = n - m;
    }
}


template <typename MatrixType, typename VectorType>
BoundaryType Problem<MatrixType, VectorType>::mpsTypeToBoundaryType(
    const char bound_type_name
) const
{
    static const std::unordered_map<char, BoundaryType> bound_type_map = {
        {'E', BoundaryType::Fixed},
        {'N', BoundaryType::Free},
        {'R', BoundaryType::Boxed},
        {'L', BoundaryType::Upper},
        {'G', BoundaryType::Lower},
    };
    
    auto it = bound_type_map.find(bound_type_name);
    if (it != bound_type_map.end()) {
        return it->second;
    }
    return BoundaryType::UNKNOWN;
}