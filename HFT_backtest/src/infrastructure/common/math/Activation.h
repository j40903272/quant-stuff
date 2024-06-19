#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "infrastructure/common/math/Math.h"

namespace alphaone
{
namespace activation
{
inline double tanh(double x)
{
    if (x > +18.36840152)
    {
        return +1.0;
    }
    if (x < -18.36840152)
    {
        return -1.0;
    }
    return (y_exp(2.0 * x) - 1.0) / (y_exp(2.0 * x) + 1.0);
}

inline double relu(double x)
{
    return std::max(x, 0.0);
}

inline double elu(double x, double a)
{
    return x > 0.0 ? x : a * (y_exp(x) - 1);
}

inline double selu(double x, double a = 1.67326324235437728481704,
                   double l = 1.05070098735548049341933)
{
    return l * elu(x, a);
}
}  // namespace activation
}  // namespace alphaone

#endif
