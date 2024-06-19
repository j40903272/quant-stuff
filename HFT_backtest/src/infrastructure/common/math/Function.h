#ifndef _FUNCTION_H_
#define _FUNCTION_H_

#include "infrastructure/common/math/Constant.h"
#include "infrastructure/common/math/Math.h"

namespace alphaone
{
// The probability density function of standard normal distribution
inline double NormalProbabilityDensityFunction(const double x)
{
    return y_exp(-0.5 * x * x) / math::MATH_SQRT_2_PI;
}

// The cumulative distribution function of standard normal distribution
// with error function calculated by std::erf (****BENCHMARK****)
inline double NormalCumulativeDistributionFunction(const double x)
{
    return 0.5 * (1.0 + std::erf(x / math::MATH_SQRT_2));
}

// The cumulative distribution function of standard normal distribution
// with error function calculated based on Hart (****FASTEST_USE_THIS****)
inline double NormalCumulativeDistributionFunctionHart(const double x)
{
    static constexpr double a1{0.0352624965998911};
    static constexpr double a2{0.700383064443688};
    static constexpr double a3{6.37396220353165};
    static constexpr double a4{33.912866078383};
    static constexpr double a5{112.079291497871};
    static constexpr double a6{221.213596169931};
    static constexpr double a7{220.206867912376};

    static constexpr double b1{0.0883883476483184};
    static constexpr double b2{1.75566716318264};
    static constexpr double b3{16.064177579207};
    static constexpr double b4{86.7807322029461};
    static constexpr double b5{296.564248779674};
    static constexpr double b6{637.333633378831};
    static constexpr double b7{793.826512519948};
    static constexpr double b8{440.413735824752};

    double z{0.0};

    const double y{std::abs(x)};
    if (y < 37.0)
    {
        if (y < 7.07106781186547)
        {
            z = y_exp(-0.5 * y * y) *
                ((((((a1 * y + a2) * y + a3) * y + a4) * y + a5) * y + a6) * y + a7) /
                (((((((b1 * y + b2) * y + b3) * y + b4) * y + b5) * y + b6) * y + b7) * y + b8);
        }
        else
        {
            z = y_exp(-0.5 * y * y) /
                ((y + 1.0 / (y + 2.0 / (y + 3.0 / (y + 4.0 / (y + 0.65))))) * 2.506628274631);
        }
    }

    if (x > 0.0)
    {
        return 1.0 - z;
    }

    return z;
}

inline double Sign(double x)
{
    return x > 0.0 ? +1.0 : x < 0.0 ? -1.0 : 0.0;
}

inline double LogTransform(double x, double k = math::MATH_GOLDEN_RATIO)
{
    return Sign(x) * y_log(1.0 + std::abs(k * x));
}

inline double PowerTransform(double x, double k = math::MATH_GOLDEN_RATIO_CONJUGATE)
{
    return Sign(x) * y_pow(std::abs(x), k);
}
}  // namespace alphaone

#endif
