#include "parser.hpp"



int LPparser::readMps(const std::string& filename, Problem& problem)
{
    CoinMpsIO mps_reader;
    int status = mps_reader.readMps(filename.c_str());
    if (status == 0 || status == 1)
    {
        CoinPackedMatrix coeffs = *mps_reader.getMatrixByCol();
        int rows_num = coeffs.getNumRows();
        int cols_num = coeffs.getNumCols();
        Matrix A(rows_num, cols_num);
        
        for (size_t i = 0; i < coeffs.getNumRows(); i++)
            for (size_t j = 0; j < coeffs.getNumCols(); j++)
                A(i, j) = coeffs.getCoefficient(i, j);

        const double* lower_range_parsed  = mps_reader.getRowLower();
        const double* upper_range_parsed  = mps_reader.getRowUpper();
        const char*   types_parsed  = mps_reader.getRowSense();

        BoundaryTypeVector range_types(rows_num);
        BoundaryTypeVector bound_types(cols_num);

        ValuesVector range_lower(rows_num);
        ValuesVector range_upper(rows_num);

        for (size_t i = 0; i < rows_num; i++)
        {
            range_types[i] = mpsTypeToBoundaryType(types_parsed[i]);
            range_lower[i] = lower_range_parsed[i];
            range_upper[i] = upper_range_parsed[i];
        }

        const double* lower_bound_parsed  = mps_reader.getColLower();
        const double* upper_bound_parsed  = mps_reader.getColUpper();
        const double* costs_parsed        = mps_reader.getObjCoefficients();

        ValuesVector costs(cols_num);
        ValuesVector bound_lower(cols_num);
        ValuesVector bound_upper(cols_num);
        
        for (size_t i = 0; i < cols_num; i++)
        {
            bound_lower[i] = lower_bound_parsed[i];
            bound_upper[i] = upper_bound_parsed[i];
            costs[i]       = costs_parsed[i];

            if (isinf_bound(bound_lower[i]) && isinf_bound(bound_upper[i]))
                bound_types[i] = BoundaryType::Free;
            else if (isinf_bound(bound_lower[i]) && !isinf_bound(bound_upper[i]))
                bound_types[i] = BoundaryType::Upper;
            else if (!isinf_bound(bound_lower[i]) && isinf_bound(bound_upper[i]))
                bound_types[i] = BoundaryType::Lower;
            else if (!isinf_bound(bound_lower[i]) && !isinf_bound(bound_upper[i]) && bound_lower[i] == bound_upper[i])
                bound_types[i] = BoundaryType::Fixed;
            else
                bound_types[i] = BoundaryType::Boxed;
        }
        
        problem = Problem(
            bound_types,
            range_types,
            costs,
            range_lower,
            range_upper,
            bound_lower,
            bound_upper,
            A
        );
    }
    
    return status;
}


BoundaryType LPparser::mpsTypeToBoundaryType(
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