#ifndef NUMERIC_H
#define NUMERIC_H

#include <type_traits>
#include <utility>

namespace alphaone
{
namespace numeric
{
template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      std::is_signed<ARITHMETIC_TYPE>::value,
                                  bool>::type
is_negative(ARITHMETIC_TYPE value)
{
    // Check if a signed value is negative
    return value < 0;
}

template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      std::is_unsigned<ARITHMETIC_TYPE>::value,
                                  bool>::type
is_negative(ARITHMETIC_TYPE value)
{
    // Check if an unsigned value is negative (it never is)
    return false;
}

template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      std::is_signed<ARITHMETIC_TYPE>::value,
                                  bool>::type
is_positive(ARITHMETIC_TYPE value)
{
    // Check if a signed value is positive
    return value > 0;
}

template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      std::is_unsigned<ARITHMETIC_TYPE>::value,
                                  bool>::type
is_positive(ARITHMETIC_TYPE value)
{
    // Check if an unsigned value is positive (it always is)
    return true;
}

template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      std::is_signed<ARITHMETIC_TYPE>::value,
                                  ARITHMETIC_TYPE>::type
abs(ARITHMETIC_TYPE value)
{
    // Return the absolute value of a signed value
    return (value < 0) ? -value : value;
}

template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      std::is_unsigned<ARITHMETIC_TYPE>::value,
                                  ARITHMETIC_TYPE>::type
abs(ARITHMETIC_TYPE value)
{
    // Return the absolute value of an unsigned value (return the value)
    return value;
}

template <typename LHS_TYPE>
constexpr LHS_TYPE min(LHS_TYPE lhs)
{
    // Return the minimum of one input value
    return lhs;
}

template <typename LHS_TYPE, typename RHS_TYPE, typename... MORE_TYPES>
constexpr typename std::common_type<LHS_TYPE, RHS_TYPE, MORE_TYPES...>::type
min(LHS_TYPE lhs, RHS_TYPE rhs, MORE_TYPES... more)
{
    // Return the minimum of two or more input valaues
    using CommonType = typename std::common_type<LHS_TYPE, RHS_TYPE, MORE_TYPES...>::type;
    return numeric::min((static_cast<CommonType>(rhs) < static_cast<CommonType>(lhs))
                            ? static_cast<CommonType>(rhs)
                            : static_cast<CommonType>(lhs),
                        more...);
}

template <typename LHS_TYPE>
constexpr LHS_TYPE max(LHS_TYPE lhs)
{
    // Return the maximum of one input value
    return lhs;
}

template <typename LHS_TYPE, typename RHS_TYPE, typename... MORE_TYPES>
constexpr typename std::common_type<LHS_TYPE, RHS_TYPE, MORE_TYPES...>::type
max(LHS_TYPE lhs, RHS_TYPE rhs, MORE_TYPES... more)
{
    // Return the maximum of two or more input valaues
    // - Uses operator< instead of operator> because this is only operator
    //   required for keys in ordered STL containers
    using CommonType = typename std::common_type<LHS_TYPE, RHS_TYPE, MORE_TYPES...>::type;
    return numeric::max((static_cast<CommonType>(lhs) < static_cast<CommonType>(rhs))
                            ? static_cast<CommonType>(rhs)
                            : static_cast<CommonType>(lhs),
                        more...);
}

template <typename BASE_TYPE, typename EXPONENT_TYPE>
constexpr typename std::enable_if<
    std::is_arithmetic<BASE_TYPE>::value && std::is_integral<EXPONENT_TYPE>::value, BASE_TYPE>::type
exp(BASE_TYPE base, EXPONENT_TYPE exponent)
{
    // Return the base value raised to an integer exponent
    return (exponent == 0) ? 1
                           : (is_negative(exponent) ? (exp(base, exponent + 1) / base)
                                                    : (exp(base, exponent - 1) * base));
}

template <typename ARITHMETIC_TYPE>
constexpr inline typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                             !std::is_floating_point<ARITHMETIC_TYPE>::value,
                                         bool>::type
is_exactly_equal(ARITHMETIC_TYPE lhs, ARITHMETIC_TYPE rhs)
{
    return lhs == rhs;
}

template <typename ARITHMETIC_TYPE>
constexpr inline typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                             std::is_floating_point<ARITHMETIC_TYPE>::value,
                                         bool>::type
is_exactly_equal(ARITHMETIC_TYPE lhs, ARITHMETIC_TYPE rhs)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"

    return lhs == rhs;

#pragma GCC diagnostic pop
}

template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      !std::is_floating_point<ARITHMETIC_TYPE>::value,
                                  bool>::type
is_exactly_zero(ARITHMETIC_TYPE value)
{
    // Compare an integer value to exactly zero
    return value == 0;
}

template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      std::is_floating_point<ARITHMETIC_TYPE>::value,
                                  bool>::type
is_exactly_zero(ARITHMETIC_TYPE value)
{
    // Compare a floating point value to exactly zero
    // - This not very safe and you should consider using numeric::is_almost_equal instead
    // - One valid use of this function might be to prevent division by zero

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"

    static_assert(-0.0 == 0.0, "Surprise! (-0.0 != 0.0)");
    return value == 0.0;

#pragma GCC diagnostic pop
}

template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      std::is_floating_point<ARITHMETIC_TYPE>::value,
                                  bool>::type
is_almost_equal(ARITHMETIC_TYPE lhs, ARITHMETIC_TYPE rhs, ARITHMETIC_TYPE epsilon)
{
    // Compare two floating point values using an epsilon value
    // - There is no universally correct way to do this!
    // - http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    return abs(lhs - rhs) <= abs(epsilon);
}

template <typename ARITHMETIC_TYPE>
constexpr typename std::enable_if<std::is_arithmetic<ARITHMETIC_TYPE>::value &&
                                      std::is_floating_point<ARITHMETIC_TYPE>::value,
                                  bool>::type
is_almost_zero(ARITHMETIC_TYPE value, ARITHMETIC_TYPE epsilon)
{
    // Compare a floating point value to zero using an epsilon value
    // - There is no universally correct way to do this!
    // - http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    return abs(value) <= abs(epsilon);
}

template <typename ARITHMETIC_TYPE, class Container>
auto equal_range(Container &&container, ARITHMETIC_TYPE value, ARITHMETIC_TYPE epsilon)
    -> decltype(container.equal_range(value))
{
    // Returns a range of elements whose keys are within epsilon from the given value
    auto lower = container.lower_bound(value - epsilon);
    auto upper = container.upper_bound(value + epsilon);
    return std::make_pair(lower, upper);
}

template <typename ARITHMETIC_TYPE, class Container>
bool key_exists(const Container &container, ARITHMETIC_TYPE value, ARITHMETIC_TYPE epsilon)
{
    // Tests for existence of a key within epsilon from the given value
    auto range = equal_range(container, value, epsilon);
    return range.first != range.second;
}

}  // namespace numeric
}  // namespace alphaone

#endif
