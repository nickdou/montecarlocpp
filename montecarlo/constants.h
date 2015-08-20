//
//  constants.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#ifndef montecarlocpp_constants_h
#define montecarlocpp_constants_h

#include <limits>
#include <cmath>

typedef std::numeric_limits<double> Dbl;
    
const double PI   = 3.141592653589793;
const double HBAR = 1.054560652927e-034;
const double KB   = 1.380648e-023;

template<typename Scalar>
bool isApprox(const Scalar& a, const Scalar& b)
{
    typedef std::numeric_limits<Scalar> Lim;
    Scalar tol = 100 * Lim::epsilon();
    Scalar max = std::max(std::abs(a), std::abs(b));
    Scalar rhs = std::max(Lim::min(), tol * max);
    return std::abs(a - b) <= rhs;
}

#endif
