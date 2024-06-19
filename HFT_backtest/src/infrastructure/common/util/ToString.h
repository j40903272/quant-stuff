#ifndef TOSTRING_H
#define TOSTRING_H

#include "infrastructure/common/numeric/Numeric.h"
#include "infrastructure/common/util/Branch.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <ratio>
#include <string>
#include <tuple>
#include <type_traits>

namespace alphaone
{
template <typename T>
std::string to_succinct_string(const T &t)
{
    std::string str{std::to_string(t)};
    size_t      offset{1};
    if (str.find('.') != std::string::npos)
    {
        if (str.find_last_not_of('0') == str.find('.'))
        {
            offset = 0;
        }
        str.erase(str.find_last_not_of('0') + offset, std::string::npos);
    }
    return str;
}

class FixedPrecisionBase;
class ToString;

namespace to_string
{
// Minimum field width (and associated fill character)
class Width
{
  public:
    constexpr Width();
    template <typename WIDTH_TYPE>
    constexpr Width(WIDTH_TYPE width,
                    typename std::enable_if<std::is_integral<WIDTH_TYPE>::value>::type * = nullptr);
    template <typename WIDTH_TYPE>
    constexpr Width(WIDTH_TYPE width, char fill,
                    typename std::enable_if<std::is_integral<WIDTH_TYPE>::value>::type * = nullptr);
    constexpr Width(const Width &) = default;
    Width &operator=(const Width &) = default;
    ~Width()                        = default;

  private:
    friend class alphaone::ToString;

    uint32_t min_width_;
    char     fill_;
};
}  // namespace to_string

constexpr to_string::Width::Width()
    : min_width_{0}  // Default width (none)
    , fill_{'0'}
{
}

template <typename WIDTH_TYPE>
constexpr to_string::Width::Width(
    WIDTH_TYPE width, typename std::enable_if<std::is_integral<WIDTH_TYPE>::value>::type *)
    : min_width_{static_cast<uint32_t>(numeric::abs(width))}, fill_{'0'}
{
}

template <typename WIDTH_TYPE>
constexpr to_string::Width::Width(
    WIDTH_TYPE width, char fill,
    typename std::enable_if<std::is_integral<WIDTH_TYPE>::value>::type *)
    : min_width_{static_cast<uint32_t>(numeric::abs(width))}, fill_{fill}
{
}

static_assert(std::is_default_constructible<to_string::Width>::value, "to_string::Width is not a literal");

namespace to_string
{
// Number of digits after the decimal point (valid range 0-18)
class Decimals
{
  public:
    constexpr Decimals();

    template <typename COUNT_TYPE>
    constexpr Decimals(
        COUNT_TYPE decimals,
        typename std::enable_if<std::is_integral<COUNT_TYPE>::value>::type * = nullptr);

    constexpr Decimals(const Decimals &) = default;

    Decimals &operator=(const Decimals &) = default;

    ~Decimals() = default;

  private:
    friend class alphaone::ToString;
    uint32_t decimals_;
};

}  // namespace to_string

constexpr to_string::Decimals::Decimals() : decimals_{6}  // Default precision
{
}

template <typename COUNT_TYPE>
constexpr to_string::Decimals::Decimals(
    COUNT_TYPE decimals, typename std::enable_if<std::is_integral<COUNT_TYPE>::value>::type *)
    : decimals_(static_cast<uint32_t>(numeric::abs(decimals)))
{
}

static_assert(std::is_default_constructible<to_string::Decimals>::value,
              "to_string::Decimals is not a literal");

namespace to_string
{
// Helper for std::enable_if<VALUE_TYPE>

template <typename TYPE>
using IsFixedPrecision = std::is_base_of<FixedPrecisionBase, TYPE>;

template <typename VALUE_TYPE>
struct CanFormat : public std::conditional<(std::is_integral<VALUE_TYPE>::value ||
                                            std::is_floating_point<VALUE_TYPE>::value ||
                                            IsFixedPrecision<VALUE_TYPE>::value),
                                           std::true_type, std::false_type>::type
{
};
}  // namespace to_string

// C++ iomanip utility to format basic arithmetic types as text
// - Can be used with ostream operator<<
// - May be useful in non-ostream contexts as well
// - Faster and more flexible than sprintf, default operator<<, etc
// - Supports minimum field width with user specified fill character
// - Formats text into an internal buffer (no heap memory required)
class ToString
{
  public:
    template <typename TYPE>
    using CanFormat = to_string::CanFormat<TYPE>;

    using Width    = to_string::Width;     // Minimum field width type
    using Decimals = to_string::Decimals;  // Decimals after radix type

    template <typename INTEGER_TYPE>
    ToString(INTEGER_TYPE integer,
             typename std::enable_if<std::is_integral<INTEGER_TYPE>::value>::type * = nullptr);

    template <typename INTEGER_TYPE>
    ToString(INTEGER_TYPE integer, Width width,
             typename std::enable_if<std::is_integral<INTEGER_TYPE>::value>::type * = nullptr);

    template <typename INTEGER_TYPE>
    ToString(INTEGER_TYPE integer, Decimals precision,
             typename std::enable_if<std::is_integral<INTEGER_TYPE>::value>::type * = nullptr);

    template <typename INTEGER_TYPE>
    ToString(INTEGER_TYPE integer, Width width, Decimals precision,
             typename std::enable_if<std::is_integral<INTEGER_TYPE>::value>::type * = nullptr);

    template <typename FLOATING_TYPE>
    ToString(
        FLOATING_TYPE floating,
        typename std::enable_if<std::is_floating_point<FLOATING_TYPE>::value>::type * = nullptr);

    template <typename FLOATING_TYPE>
    ToString(
        FLOATING_TYPE floating, Width width,
        typename std::enable_if<std::is_floating_point<FLOATING_TYPE>::value>::type * = nullptr);

    template <typename FLOATING_TYPE>
    ToString(
        FLOATING_TYPE floating, Decimals precision,
        typename std::enable_if<std::is_floating_point<FLOATING_TYPE>::value>::type * = nullptr);

    template <typename FLOATING_TYPE>
    ToString(
        FLOATING_TYPE floating, Width width, Decimals precision,
        typename std::enable_if<std::is_floating_point<FLOATING_TYPE>::value>::type * = nullptr);

    template <typename FIXED_TYPE>
    ToString(
        FIXED_TYPE fixed,
        typename std::enable_if<to_string::IsFixedPrecision<FIXED_TYPE>::value>::type * = nullptr);

    template <typename FIXED_TYPE>
    ToString(
        FIXED_TYPE fixed, Width width,
        typename std::enable_if<to_string::IsFixedPrecision<FIXED_TYPE>::value>::type * = nullptr);

    template <typename FIXED_TYPE>
    ToString(
        FIXED_TYPE fixed, Decimals precision,
        typename std::enable_if<to_string::IsFixedPrecision<FIXED_TYPE>::value>::type * = nullptr);

    template <typename FIXED_TYPE>
    ToString(
        FIXED_TYPE fixed, Width width, Decimals precision,
        typename std::enable_if<to_string::IsFixedPrecision<FIXED_TYPE>::value>::type * = nullptr);

    ToString(const ToString &) = delete;

    ToString &operator=(const ToString &) = delete;

    ~ToString() = default;

    // Access the result string

    const char *c_str() const;

    std::string to_string() const;

    std::ostream &to_ostream(std::ostream &output) const;

  private:
    // Absolute worst case output size is ...
    // - sign character
    // - 20 digits (max uint64_t)
    // - decimal point
    // - 20 digits (max uint64_t)
    // - nullptr character
    // = 43 bytes
    // - Round up to 48 bytes (an 8-byte boundary)

    static constexpr size_t   _buffer_size();
    static constexpr uint32_t _max_width();

    template <typename INTEGER_TYPE>
    static char *_integer_to_string(bool         negative,      // Force negative sign?
                                    INTEGER_TYPE integer,       // Integer input
                                    Width        width,         // Minimum field width
                                    char         terminator,    // Termination character
                                    char *       write_begin,   // Write area begin
                                    char *       write_end,     // Write area end
                                    bool         chomp = false  // Chomp trailing zeros?
    );

    template <typename FLOATING_TYPE>
    void _floating_to_string(FLOATING_TYPE floating,
                             Width         width,     // Minimum field width
                             Decimals      precision  // Decimal precision
    );

    template <typename FIXED_TYPE>
    void _fixed_to_string(FIXED_TYPE fixed,
                          Width      width,     // Minimum field width
                          Decimals   precision  // Decimal precision
    );

    char  _buffer[48];  // Same as _buffer_size()
    char *_front;       // First valid result character
};

template <typename INTEGER_TYPE>
inline ToString::ToString(INTEGER_TYPE integer,
                          typename std::enable_if<std::is_integral<INTEGER_TYPE>::value>::type *)
    : _buffer()
    , _front(_integer_to_string(false,  // Determine negative sign in method
                                integer,
                                Width(),  // No minimum field width, terminator ignored
                                '\0',     // nullptr terminate string
                                static_cast<char *>(_buffer),
                                static_cast<char *>(_buffer) + _buffer_size()))
{
}

template <typename INTEGER_TYPE>
inline ToString::ToString(INTEGER_TYPE integer, Width width,
                          typename std::enable_if<std::is_integral<INTEGER_TYPE>::value>::type *)
    : _buffer()
    , _front(_integer_to_string(
          false,  // Determine negative sign in method
          integer, Width(numeric::min(_max_width(), width.min_width_), width.fill_),
          '\0',  // nullptr terminate string
          static_cast<char *>(_buffer), static_cast<char *>(_buffer) + _buffer_size()))
{
}

template <typename INTEGER_TYPE>
inline ToString::ToString(INTEGER_TYPE integer, Decimals precision,
                          typename std::enable_if<std::is_integral<INTEGER_TYPE>::value>::type *)
    : _buffer()
    , _front(_integer_to_string(false,  // Determine negative sign in method
                                integer,
                                Width(),  // No minimum field width, terminator ignored
                                '\0',     // nullptr terminate string
                                static_cast<char *>(_buffer),
                                static_cast<char *>(_buffer) + _buffer_size()))
{
}

template <typename INTEGER_TYPE>
inline ToString::ToString(INTEGER_TYPE integer, Width width, Decimals precision,
                          typename std::enable_if<std::is_integral<INTEGER_TYPE>::value>::type *)
    : _buffer()
    , _front(_integer_to_string(
          false,  // Determine negative sign in method
          integer, Width(numeric::min(_max_width(), width.min_width_), width.fill_),
          '\0',  // nullptr terminate string
          static_cast<char *>(_buffer), static_cast<char *>(_buffer) + _buffer_size()))
{
}

template <typename FLOATING_TYPE>
inline ToString::ToString(
    FLOATING_TYPE floating,
    typename std::enable_if<std::is_floating_point<FLOATING_TYPE>::value>::type *)
    : _buffer(), _front(nullptr)
{
    _floating_to_string(floating,
                        Width(),    // No minimum field width, terminator ignored
                        Decimals()  // Default precision
    );
}

template <typename FLOATING_TYPE>
inline ToString::ToString(
    FLOATING_TYPE floating, Width width,
    typename std::enable_if<std::is_floating_point<FLOATING_TYPE>::value>::type *)
    : _buffer(), _front(nullptr)
{
    _floating_to_string(floating,
                        width,      // User provided minimum field width
                        Decimals()  // Default precision
    );
}

template <typename FLOATING_TYPE>
inline ToString::ToString(
    FLOATING_TYPE floating, Decimals precision,
    typename std::enable_if<std::is_floating_point<FLOATING_TYPE>::value>::type *)
    : _buffer(), _front(nullptr)
{
    _floating_to_string(floating,
                        Width(),   // No minimum field width, terminator ignored
                        precision  // User provided precision
    );
}

template <typename FLOATING_TYPE>
inline ToString::ToString(
    FLOATING_TYPE floating, Width width, Decimals precision,
    typename std::enable_if<std::is_floating_point<FLOATING_TYPE>::value>::type *)
    : _buffer(), _front(nullptr)
{
    _floating_to_string(floating,
                        width,     // User provided minimum field width
                        precision  // User provided precision
    );
}
template <typename FIXED_TYPE>
inline ToString::ToString(
    FIXED_TYPE fixed,
    typename std::enable_if<to_string::IsFixedPrecision<FIXED_TYPE>::value>::type *)
    : _buffer(), _front(nullptr)
{
    _fixed_to_string(fixed,
                     Width(),  // No minimum field width, terminator ignored
                     Decimals((FIXED_TYPE::template exponent_mantissa<>::exponent < 0)
                                  ? -FIXED_TYPE::template exponent_mantissa<>::exponent
                                  : 0  // Integer precision (no decimals)
                              ));
}

template <typename FIXED_TYPE>
inline ToString::ToString(
    FIXED_TYPE fixed, Width width,
    typename std::enable_if<to_string::IsFixedPrecision<FIXED_TYPE>::value>::type *)
    : _buffer(), _front(nullptr)
{
    _fixed_to_string(fixed,
                     width,  // User provided minimum field width
                     Decimals((FIXED_TYPE::template exponent_mantissa<>::exponent < 0)
                                  ? -FIXED_TYPE::template exponent_mantissa<>::exponent
                                  : 0  // Integer precision (no decimals)
                              ));
}

template <typename FIXED_TYPE>
inline ToString::ToString(
    FIXED_TYPE fixed, Decimals precision,
    typename std::enable_if<to_string::IsFixedPrecision<FIXED_TYPE>::value>::type *)
    : _buffer(), _front(nullptr)
{
    // !!! IMPORTANT !!! IMPORTANT !!! IMPORTANT !!! IMPORTANT !!!
    //
    // Ignore user provided precision, it is presently broken for
    // fixed precision types
    //
    // !!! IMPORTANT !!! IMPORTANT !!! IMPORTANT !!! IMPORTANT !!!

    _fixed_to_string(fixed,
                     Width(),  // No minimum field width, terminator ignored
                     Decimals((FIXED_TYPE::template exponent_mantissa<>::exponent < 0)
                                  ? -FIXED_TYPE::template exponent_mantissa<>::exponent
                                  : 0  // Integer precision (no decimals)
                              ));
}

template <typename FIXED_TYPE>
inline ToString::ToString(
    FIXED_TYPE fixed, Width width, Decimals precision,
    typename std::enable_if<to_string::IsFixedPrecision<FIXED_TYPE>::value>::type *)
    : _buffer(), _front(nullptr)
{
    // !!! IMPORTANT !!! IMPORTANT !!! IMPORTANT !!! IMPORTANT !!!
    //
    // Ignore user provided precision, it is presently broken for
    // fixed precision types
    //
    // !!! IMPORTANT !!! IMPORTANT !!! IMPORTANT !!! IMPORTANT !!!

    _fixed_to_string(fixed,
                     width,  // User provided minimum field width
                     Decimals((FIXED_TYPE::template exponent_mantissa<>::exponent < 0)
                                  ? -FIXED_TYPE::template exponent_mantissa<>::exponent
                                  : 0  // Integer precision (no decimals)
                              ));
}

constexpr size_t ToString::_buffer_size()
{
    static_assert(sizeof(_buffer) == 48, "Buffer size error");
    return 48;
}

constexpr uint32_t ToString::_max_width()
{
    static_assert(sizeof(_buffer) > 40, "Buffer size error");
    return 40;
}

template <typename INTEGER_TYPE>
inline char *ToString::_integer_to_string(bool         negative,     // Force negative sign?
                                          INTEGER_TYPE integer,      // Integer input
                                          Width        width,        // Minimum field width
                                          char         terminator,   // Termination character
                                          char *       write_begin,  // Write area begin
                                          char *       write_end,    // Write area end
                                          bool         chomp         // Chmop trailing zeros?
)
{
    // ASSUMPTIONS:
    // - Inputs to this private helper function are assumed to be valid!
    // - The write area must be large enough to hold the integer string!

    // Output at least one digit if the input integer is zero

    if ((integer == 0) && chomp)
    {
        write_begin[0] = '0';
        write_begin[1] = terminator;

        return write_begin;
    }

    // Convert digits from right to left (least to most significant)

    char *write = --write_end;

    // Apply the termination character (do not count against min width)

    *write = terminator;
    --write;

    // Check for a negative integer

    bool    is_negative = negative;
    int32_t min_width   = static_cast<int32_t>(width.min_width_);

    if (numeric::is_negative(integer))
    {
        --min_width;
        integer     = -integer;
        is_negative = true;
    }

    // Output at least one digit if the input integer is zero

    if (integer == 0)
    {
        *write = '0';
        --write;
        --min_width;
    }

    // Convert the integer to a string, working from right to left

    while (chomp && (integer > 0))
    {
        char digit = (integer % 10) + '0';

        if (digit == '0')
        {
            *write = '\0';
        }
        else
        {
            *write = digit;
            chomp  = false;
        }

        --write;
        --min_width;
        integer /= 10;
    }

    while (integer > 0)
    {
        *write = (integer % 10) + '0';
        --write;
        --min_width;
        integer /= 10;
    }

    if (is_negative && (width.fill_ != '0'))
    {
        *write = '-';  // Output '-' before digits, after filler
        --write;
    }

    while (min_width > 0)
    {
        *write = width.fill_;
        --write;
        --min_width;
    }

    if (is_negative && (width.fill_ == '0'))
    {
        *write = '-';  // Output '-' before '0' digits
        --write;
    }

    // Return a pointer to the beginning of the output string

    return ++write;
}

template <typename FLOATING_TYPE>
inline void ToString::_floating_to_string(FLOATING_TYPE floating, Width width, Decimals precision)
{
    // Apply max width constant to prevent buffer overflow

    const uint32_t min_width = numeric::min(_max_width(), width.min_width_);

    // Calculate fractional part denominator

    const double denominator = numeric::exp(10.0, numeric::abs(precision.decimals_));

    // Split the floating point value into whole and fractional parts

    bool     negative;
    uint64_t whole;
    uint64_t fraction;

    if (floating >= 0.0)
    {
        negative = false;
        whole    = static_cast<uint64_t>(floating);
        fraction = static_cast<uint64_t>(
            (floating - static_cast<FLOATING_TYPE>(whole)) * denominator + 0.5);
    }
    else
    {
        negative = true;
        whole    = static_cast<uint64_t>(-floating);
        fraction = static_cast<uint64_t>(
            (-floating - static_cast<FLOATING_TYPE>(whole)) * denominator + 0.5);
    }

    // Convert the whole part of the value

    uint32_t whole_width;

    if (min_width > (precision.decimals_ + 1))
    {
        whole_width = min_width - precision.decimals_ - 1 - (negative ? 1 : 0);
    }
    else
    {
        whole_width = 0;
    }

    char *decimal_point = _buffer + (_buffer_size() / 2);

    _front = _integer_to_string(negative, whole, Width(whole_width, width.fill_),
                                '.',  // Decimal point terminate string
                                static_cast<char *>(_buffer), decimal_point + 1);

    const std::size_t whole_part_length = decimal_point - _front;

    // Convert the fractional part of the value

    _integer_to_string(false, fraction, Width(precision.decimals_, '0'),
                       '\0',  // nullptr terminate string
                       decimal_point + 1, decimal_point + 2 + precision.decimals_, min_width == 0);

    // Apply width (may truncate fractional part of the value)

    if ((min_width > 0) && (min_width > whole_part_length))
    {
        _front[min_width] = '\0';
    }
}

template <typename FIXED_TYPE>
inline void ToString::_fixed_to_string(FIXED_TYPE fixed, Width width, Decimals precision)
{
    if ((precision.decimals_ == 0) && (FIXED_TYPE::template exponent_mantissa<>::exponent <= 0))
    {
        // We have not been asked to output digits after the decimal point,
        // and the underlying type is integral or greater.

        _front = _integer_to_string(false,  // Determine negative sign in method
                                    fixed.template to<std::ratio<1>>(), width,
                                    '\0',  // nullptr terminate string
                                    static_cast<char *>(_buffer),
                                    static_cast<char *>(_buffer) + _buffer_size());

        return;
    }

    // Apply max width constant to prevent buffer overflow

    const uint32_t min_width = numeric::min(_max_width(), width.min_width_);

    // Split the fixed value into whole and fractional parts

    bool     negative;
    uint64_t whole;
    uint64_t fraction;

    std::tie(negative, whole, fraction) = fixed.template to_split<typename FIXED_TYPE::ratio>();

    // Convert the whole part of the value

    uint32_t whole_width;

    if (min_width > (precision.decimals_ + 1))
    {
        whole_width = min_width - precision.decimals_ - 1 - (negative ? 1 : 0);
    }
    else
    {
        whole_width = 0;
    }

    char *decimal_point = _buffer + (_buffer_size() / 2);

    _front = _integer_to_string(negative, whole, Width(whole_width, width.fill_),
                                '.',  // Decimal point terminate string
                                static_cast<char *>(_buffer), decimal_point + 1);

    const std::size_t whole_part_length = decimal_point - _front;

    // Our fraction might be too big

    for (int i = -FIXED_TYPE::template exponent_mantissa<>::exponent; i > int(precision.decimals_);
         i--)
    {
        fraction = fraction / 10 + (fraction % 10 / 5);
    }

    // Convert the fractional part of the value

    _integer_to_string(false, fraction, Width(precision.decimals_, '0'),
                       '\0',  // nullptr terminate string
                       decimal_point + 1, decimal_point + 2 + precision.decimals_, min_width == 0);

    // Apply width (may truncate fractional part of the value)

    if ((min_width > 0) && (min_width > whole_part_length))
    {
        _front[min_width] = '\0';
    }
}

inline const char *ToString::c_str() const
{
    return _front;
}

inline std::string ToString::to_string() const
{
    return std::string(_front);
}

inline std::ostream &ToString::to_ostream(std::ostream &output) const
{
    return (output << _front);
}

inline std::ostream &operator<<(std::ostream &output, const ToString &to_string)
{
    return to_string.to_ostream(output);
}

}  // namespace alphaone

#endif
