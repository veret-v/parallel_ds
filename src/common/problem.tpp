#include "problem.hpp"


template <typename MatrixType, typename VectorType>
Problem<MatrixType, VectorType>::Problem(
    const Problem& problem
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

    scale_rows = VectorType(constraints_size);
    scale_cols = VectorType(problem_size);

    solution.Z = 0;
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
    const Matrix& _A
) : 
bound_type(_bound_type), 
costs(_costs), 
upper_bound(_upper_bound), 
lower_bound(_lower_bound)
{
    VectorType lower_range = _lower_range;
    VectorType upper_range = _upper_range;
    BoundaryTypeVector range_type = _range_type;
    Matrix A_buff = _A;

    reduce(lower_range, upper_range, range_type, A_buff);
    if (!checkWellScaled(A_buff)) scale(A_buff, lower_range, upper_range);
    transformToComputeForm(lower_range, upper_range, range_type, A_buff);
    
    A = A_buff;
    checkConstraints();
    
    int m = std::get<0>(this->A.getSize());
    int n = std::get<1>(this->A.getSize());

    this -> problem_size = n;
    this -> constraints_size = m;
    this -> logicals_size = n - m;

    solution.Z = 0;
}

template <typename MatrixType, typename VectorType>
void Problem<MatrixType, VectorType>::transformToComputeForm(
    const VectorType &lower_range,
    const VectorType &upper_range,
    const BoundaryTypeVector &range_type,
    Matrix& A_buff
)
{
    int m = std::get<0>(A_buff.getSize());
    int n = std::get<1>(A_buff.getSize());
    A_buff.stackColUnitMatrix();

    #ifdef WITH_CUDA
        if constexpr (std::is_same_v<VectorType, CudaDataDenseVector>) {
            upper_bound.resize(m + n);
            lower_bound.resize(m + n);
            costs.resize(m + n);
        }  
    #endif

    for (int i = 0; i < m; i++)
    {
        #ifndef WITH_CUDA
            upper_bound.pushBack(-lower_range[i]);
            lower_bound.pushBack(-upper_range[i]);
            costs.pushBack(0);
        #else
            if constexpr (std::is_same_v<VectorType, CudaDataDenseVector>) {
                upper_bound[n + i] = -lower_range[i];
                lower_bound[n + i] = -upper_range[i];
                costs[n + i] = 0;
            } else {
                upper_bound.pushBack(-lower_range[i]);
                lower_bound.pushBack(-upper_range[i]);
                costs.pushBack(0);
            }
        #endif
        
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

/* strange function....*/
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
void Problem<MatrixType, VectorType>::reduce(
    VectorType &lower_range,
    VectorType &upper_range,
    BoundaryTypeVector &range_type,
    Matrix& A_buff
)
{
    double* elems = A_buff.getNonZeroElems();
    int* row_ptrs = A_buff.getRowPtrs();
    int* col_ids  = A_buff.getColIds();

    int elems_size = A_buff.getNonZeroSize();
    int m, n;

    std::set<int> fixed_vars;

    std::tie(m, n) = A_buff.getSize();

    for (int i = 0; i < m; i++)
    {
        for (int j = row_ptrs[i]; j < row_ptrs[i + 1]; j++)
        {
            int col_num = col_ids[j];
            if (bound_type[col_num] == BoundaryType::Fixed)
            {
                fixed_vars.insert(col_num);
                lower_range[i] -= elems[j] * lower_bound[col_num];
                upper_range[i] -= elems[j] * lower_bound[col_num];
            }
        }
    }

    BoundaryTypeVector new_bound_type;
    for (int i = 0; i < bound_type.size(); i++)
    {
        if (fixed_vars.find(i) == fixed_vars.end()) 
            new_bound_type.push_back(bound_type[i]);
        else
            solution.Z += lower_bound[i] * costs[i];
    }
    bound_type = new_bound_type;
    

    std::set<int> except_row =  A_buff.deleteCols(fixed_vars);

    costs.deleteVals(fixed_vars);
    lower_bound.deleteVals(fixed_vars);
    upper_bound.deleteVals(fixed_vars);
    lower_range.deleteVals(except_row);
    upper_range.deleteVals(except_row);

    BoundaryTypeVector new_range_type;
    for (int i = 0; i < range_type.size(); i++)
    {
        if (except_row.find(i) == except_row.end()) 
            new_range_type.push_back(range_type[i]);
    }
    range_type = new_range_type;
}


template <typename MatrixType, typename VectorType>
void Problem<MatrixType, VectorType>::scale(
    Matrix& A_buff,
    VectorType &lower_range,
    VectorType &upper_range
)
{
    scaled = true;

    double* elems = A_buff.getNonZeroElems();
    int* row_ptrs = A_buff.getRowPtrs();
    int* col_ids  = A_buff.getColIds();
    int elems_size = A_buff.getNonZeroSize();
    int m, n;

    std::tie(m, n) = A_buff.getSize();

    scale_rows = VectorType(m);
    scale_cols = VectorType(n);

    for (int i = 0; i < m; i++)
    {
        double max_col = fabs(elems[row_ptrs[i]]);
        for (int j = row_ptrs[i]; j < row_ptrs[i + 1]; j++)
        {
            if (max_col < fabs(elems[j]))
                max_col = fabs(elems[j]);
        }

        scale_rows[i] = max_col;
        lower_range[i] = lower_range[i] / max_col;
        upper_range[i] = upper_range[i] / max_col;

        for (int j = row_ptrs[i]; j < row_ptrs[i + 1]; j++)
            elems[j] =  elems[j] / max_col;

        for (int j = row_ptrs[i]; j < row_ptrs[i + 1]; j++)
        {
            if (scale_cols[col_ids[j]] < fabs(elems[j]))
                scale_cols[col_ids[j]] = fabs(elems[j]);
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = row_ptrs[i]; j < row_ptrs[i + 1]; j++)
            elems[j] = scale_cols[col_ids[j]] ?  elems[j] / scale_cols[col_ids[j]] : elems[j];
    }

    for (int i = 0; i < n; i++)
    {
        costs[i] = scale_cols[i] ? costs[i] / scale_cols[i] : costs[i] ;
        upper_bound[i] = scale_cols[i] ? upper_bound[i] * scale_cols[i] : upper_bound[i];
        lower_bound[i] = scale_cols[i] ? lower_bound[i] * scale_cols[i] : lower_bound[i];
    }
    
}


template <typename MatrixType, typename VectorType>
LPsolution Problem<MatrixType, VectorType>::getSolution()
{
    if (scaled)
    {
        ValuesVector original_costs(problem_size);
        for (size_t i = 0; i < problem_size - constraints_size; i++)
        {
            solution.x[i] = scale_cols[i] ? solution.x[i] / scale_cols[i] : solution.x[i];
            original_costs[i] = scale_cols[i] ? costs[i] * scale_cols[i] : costs[i];
        }
        solution.Z = original_costs.dot(solution.x);
    }
    return solution;
}


template <typename MatrixType, typename VectorType>
bool Problem<MatrixType, VectorType>::checkWellScaled(Matrix& A_buff)
{
    double* elems = A_buff.getNonZeroElems();
    int elems_size = A_buff.getNonZeroSize();

    double max_elem = (elems_size) ? fabs(elems[0]) : 0;
    double min_elem = (elems_size) ? fabs(elems[0]) : 0;
    for (int i = 0; i < elems_size; i++)
    {
        if (min_elem > fabs(elems[i]))
            min_elem = fabs(elems[i]);

        if (max_elem < fabs(elems[i]))
            max_elem = fabs(elems[i]);
    }
    return std::log10(max_elem / min_elem) < 2;
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
    // A.show();
    std::cout << "rhs vector:" << std::endl;
    RHS.show();
    std::cout << "boundary:" << std::endl;
    for (int i = 0; i < bound_type.size(); i++)
    {
        std::cout << boundaryTypeToString(bound_type[i]) << "  " << lower_bound[i] << "  " << upper_bound[i] <<std::endl;
    }
}


template <typename MatrixType, typename VectorType>
void Problem<MatrixType, VectorType>::readMps(const std::string& filename)
{
    CoinMpsIO mps_reader;
    int status = mps_reader.readMps(filename.c_str());
    if (status == 0 || status == 1)
    {
        CoinPackedMatrix coeffs = *mps_reader.getMatrixByRow();

        int rows_num = coeffs.getNumRows();
        int cols_num = coeffs.getNumCols();

        Matrix A_buff = coeffs;
       
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

        reduce(range_lower, range_upper, range_types, A_buff);
        if (!checkWellScaled(A_buff)) scale(A_buff, range_lower, range_upper);
        transformToComputeForm(range_lower, range_upper, range_types, A_buff);
        /*
            WRITE MOVE CONSTRUCTORS
        */

        A = A_buff;
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
