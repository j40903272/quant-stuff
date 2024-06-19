#ifndef _DATE_H_
#define _DATE_H_

#include "infrastructure/common/util/ToString.h"

// clang-format off
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/spdlog/fmt/ostr.h"
// clang-format on

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>
#include <ratio>
#include <sstream>
#include <string>
#include <type_traits>

namespace boost
{
namespace gregorian
{
class date;
}
}  // namespace boost

namespace alphaone
{
class Date
{
    // representation of a date
    // (stored in 4 bytes as a year, month, and day)

  public:
    constexpr Date();
    constexpr Date(const Date &) = default;
    Date &operator=(const Date &) = default;
    ~Date()                       = default;

    // constants
    static constexpr Date invalid_date();
    static constexpr Date min_date();
    static constexpr Date max_date();

    // get current date
    static Date today();

    // convert from another format
    static constexpr Date from_yyyymmdd(int32_t yyyymmdd);
    static constexpr Date from_year_month_day(int16_t yyyy, int8_t mm, int8_t dd);
    static Date           from_date_str(const char *str);  // from 'yyyymmdd' NOT 'yyyy-mm-dd'
    static Date           from_gregorian_date(const boost::gregorian::date &date);

    // convert to another date format
    constexpr int32_t year() const;
    constexpr int32_t month() const;
    constexpr int32_t day() const;
    constexpr int32_t to_yyyymmdd() const;
    void              to_gregorian_date(boost::gregorian::date &date) const;
    std::string       to_month_name() const;

    // convert to string format
    std::string   to_string() const;            // ex: "20180713"
    std::string   to_string_with_dash() const;  // ex: "2018-07-13"
    std::ostream &to_ostream(std::ostream &output) const;

    // check date validity
    // (true if time is within non-inclusive min_time/max_time range)
    explicit constexpr operator bool() const;

    // comparison operators
    constexpr bool operator==(Date rhs) const;
    constexpr bool operator!=(Date rhs) const;
    constexpr bool operator<(Date rhs) const;
    constexpr bool operator<=(Date rhs) const;
    constexpr bool operator>(Date rhs) const;
    constexpr bool operator>=(Date rhs) const;

    // takes the difference between two dates
    constexpr Date operator+(int32_t days) const;
    constexpr Date operator-(int32_t days) const;

    Date &operator+=(int32_t days);
    Date &operator++();
    Date  operator++(int);

    Date &operator-=(int32_t days);
    Date &operator--();
    Date  operator--(int);

    constexpr int32_t operator-(const Date &rhs) const;

    // returns the day of the week, with Sunday starting at 0
    constexpr uint8_t day_of_week() const;

    // returns the week number for this date, counting from year 0
    constexpr uint32_t week() const;

    // gets the next friday, including today
    Date get_next_friday() const;

    // gets the next last friday of the month, including today
    Date get_next_last_friday_of_the_month() const;

    // gets the next last friday of the quarter, including today
    Date get_next_last_friday_of_the_quarter() const;

    template <typename OStream>
    friend OStream &operator<<(OStream &os, const Date &d)
    {
        return os << "[" << d.to_string_with_dash() << "]";
    }

  private:
    class DateValue;

    explicit constexpr Date(int16_t yyyy, int8_t mm, int8_t dd);

    static std::map<int32_t, std::string> to_month_name_;

    int16_t yyyy_;
    int8_t  mm_;
    int8_t  dd_;
};

class Date::DateValue
{
    // helper class for date calculations

  public:
    constexpr DateValue(const Date &);
    constexpr DateValue(const DateValue &) = default;
    DateValue &operator=(const DateValue &) = default;
    ~DateValue()                            = default;

    constexpr DateValue from_value(int32_t v) const;
    constexpr Date      to_date() const;
    constexpr int32_t   value() const;
    constexpr DateValue operator+(int32_t days) const;
    constexpr int32_t   operator-(const DateValue &rhs) const;
    constexpr uint8_t   day_of_week() const;
    constexpr uint32_t  week() const;

  private:
    explicit constexpr DateValue(int16_t y, int8_t m, int8_t d);
    constexpr DateValue from_value_yyyy(int32_t v, int16_t y) const;
    constexpr DateValue from_value_yyyy_mm(int32_t v, int16_t y, int8_t m) const;

    Date    date_;
    int32_t value_;
};

constexpr Date::Date() : yyyy_(0), mm_(0), dd_(0)
{
}

constexpr Date::Date(int16_t yyyy, int8_t mm, int8_t dd)  // Private!
    : yyyy_(yyyy), mm_(mm), dd_(dd)
{
}

constexpr Date Date::invalid_date()
{
    return Date(0, 0, 0);
}

constexpr Date Date::min_date()
{
    return Date(1900, 03, 01);
}

constexpr Date Date::max_date()
{
    return Date(2100, 01, 01);
}

constexpr Date Date::from_yyyymmdd(int32_t yyyymmdd)
{
    return Date(yyyymmdd / 10000, (yyyymmdd / 100) % 100, yyyymmdd % 100);
}

constexpr Date Date::from_year_month_day(int16_t yyyy, int8_t mm, int8_t dd)
{
    return Date(yyyy, mm, dd);
}

constexpr int32_t Date::year() const
{
    return yyyy_;
}

constexpr int32_t Date::month() const
{
    return static_cast<int32_t>(mm_);
}

constexpr int32_t Date::day() const
{
    return static_cast<int32_t>(dd_);
}

constexpr int32_t Date::to_yyyymmdd() const
{
    return (yyyy_ * 10000) + static_cast<int32_t>(mm_) * 100 + static_cast<int32_t>(dd_);
}

inline std::string Date::to_string() const
{
    return ToString(to_yyyymmdd()).c_str();
}

inline std::string Date::to_string_with_dash() const
{
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << static_cast<int>(yyyy_);
    ss << "-";
    ss << std::setw(2) << std::setfill('0') << static_cast<int>(mm_);
    ss << "-";
    ss << std::setw(2) << std::setfill('0') << static_cast<int>(dd_);

    return ss.str();
}

inline std::ostream &Date::to_ostream(std::ostream &output) const
{
    return (output << ToString(to_yyyymmdd()).c_str());
}

constexpr Date::operator bool() const
{
    return (*this > min_date()) && (*this < max_date());
}

constexpr bool Date::operator==(Date rhs) const
{
    return (dd_ == rhs.dd_) and (mm_ == rhs.mm_) and (yyyy_ == rhs.yyyy_);
}

constexpr bool Date::operator!=(Date rhs) const
{
    return !(*this == rhs);
}

constexpr bool Date::operator<(Date rhs) const
{
    return to_yyyymmdd() < rhs.to_yyyymmdd();
}

constexpr bool Date::operator<=(Date rhs) const
{
    return to_yyyymmdd() <= rhs.to_yyyymmdd();
}

constexpr bool Date::operator>(Date rhs) const
{
    return (rhs < *this);
}

constexpr bool Date::operator>=(Date rhs) const
{
    return (rhs <= *this);
}

constexpr Date Date::operator+(int32_t n) const
{
    return (DateValue(*this) + n).to_date();
}

constexpr Date Date::operator-(int32_t n) const
{
    return operator+(-n);
}

inline Date &Date::operator+=(int32_t n)
{
    return (*this = operator+(n));
}

inline Date &Date::operator++()
{
    return operator+=(1);
}

inline Date Date::operator++(int)
{
    Date result(*this);
    ++(*this);
    return result;
}

inline Date &Date::operator-=(int32_t n)
{
    return (*this = operator+(-n));
}

inline Date &Date::operator--()
{
    return operator+=(-1);
}

inline Date Date::operator--(int)
{
    Date result(*this);
    --(*this);
    return result;
}

constexpr int32_t Date::operator-(const Date &date) const
{
    return DateValue(*this) - DateValue(date);
}

constexpr uint8_t Date::day_of_week() const
{
    return DateValue(*this).day_of_week();
}

constexpr uint32_t Date::week() const
{
    return DateValue(*this).week();
}

inline Date Date::get_next_friday() const
{
    Date next_friday = *this;
    while (next_friday.day_of_week() != 5)
        ++next_friday;
    return next_friday;
}

inline Date Date::get_next_last_friday_of_the_month() const
{
    Date next_friday = get_next_friday();
    while (next_friday.month() == (next_friday + 7).month())
        next_friday += 7;
    return next_friday;
}

inline Date Date::get_next_last_friday_of_the_quarter() const
{
    Date next_friday = get_next_friday();
    while ((int)((next_friday.month() - 1) / 3) == (int)(((next_friday + 7).month() - 1) / 3))
        next_friday += 7;
    return next_friday;
}

inline std::ostream &operator<<(std::ostream &output, const Date &date)
{
    return date.to_ostream(output);
}

static_assert(sizeof(Date) == 4, "Date has unexpected size");
static_assert(alignof(Date) == 2, "Invalid alignment");
static_assert(std::is_default_constructible<Date>::value, "Date is not a literal");

constexpr Date::DateValue::DateValue(const Date &date)
    : date_((date.month() < 3)
                ? Date::from_year_month_day(date.year() - 1, date.month() + 9, date.day())
                : Date::from_year_month_day(date.year(), date.month() - 3, date.day()))
    , value_(date_.year() * 1461 / 4 + (date_.month() * 979 + 15) / 32 + date_.day())

{
}

constexpr Date::DateValue::DateValue(int16_t y, int8_t m, int8_t d)
    : date_(Date::from_year_month_day(y, m, d))
    , value_(date_.year() * 1461 / 4 + (date_.month() * 979 + 15) / 32 + date_.day())
{
}

constexpr int32_t Date::DateValue::value() const
{
    return value_;
}

constexpr Date::DateValue Date::DateValue::from_value(int32_t v) const
{
    return Date::DateValue::from_value_yyyy(v, (v * 8 - 2) / 2922);
}

constexpr Date::DateValue Date::DateValue::from_value_yyyy(int32_t v, int16_t y) const
{
    return Date::DateValue::from_value_yyyy_mm(v, y, (v * 32 - y * 1461 / 4 * 32 - 16) / 979);
}

constexpr Date::DateValue Date::DateValue::from_value_yyyy_mm(int32_t v, int16_t y, int8_t m) const
{
    return DateValue(y, m, v - (y * 1461 / 4 + (m * 979 + 15) / 32));
}

constexpr Date::DateValue Date::DateValue::operator+(int32_t n) const
{
    return Date::DateValue::from_value(value_ + n);
}

constexpr int32_t Date::DateValue::operator-(const Date::DateValue &rhs) const
{
    return value_ - rhs.value_;
}

constexpr Date Date::DateValue::to_date() const
{
    return date_.month() > 9
               ? Date::from_year_month_day(date_.year() + 1, date_.month() - 9, date_.day())
               : Date::from_year_month_day(date_.year(), date_.month() + 3, date_.day());
}
constexpr uint8_t Date::DateValue::day_of_week() const
{
    // a value_ of zero corresponds to a Tuesday, we add two to make Sunday
    // the first day of the week
    // Sunday -> 0, Friday -> 5
    return (value_ + 1) % 7;
}
constexpr uint32_t Date::DateValue::week() const
{
    return (value_ + 1) / 7;
}
}  // namespace alphaone

#endif
