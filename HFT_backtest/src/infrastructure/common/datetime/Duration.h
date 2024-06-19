#ifndef DURATION_H
#define DURATION_H 1

#include "infrastructure/common/util/ToString.h"

#include <ratio>  // std::nano::den
#include <sstream>
#include <string>

namespace alphaone
{
class Duration
{
  public:
    // a duration is a period of time
    // stores in nanoseconds

    constexpr Duration();
    constexpr Duration(const Duration &) = default;
    Duration &operator=(const Duration &) = default;
    ~Duration()                           = default;

    // constants
    static constexpr Duration zero();
    static constexpr Duration one();
    static constexpr Duration min_duration();
    static constexpr Duration max_duration();

    // convert from another time unit
    static constexpr Duration from_hour(int64_t hour);
    static constexpr Duration from_min(int64_t min);
    static constexpr Duration from_sec(int64_t sec);
    static constexpr Duration from_msec(int64_t msec);
    static constexpr Duration from_usec(int64_t usec);
    static constexpr Duration from_nsec(int64_t nsec);

    static constexpr Duration from_hour(double hour);
    static constexpr Duration from_min(double min);
    static constexpr Duration from_sec(double sec);
    static constexpr Duration from_msec(double msec);
    static constexpr Duration from_usec(double usec);
    static constexpr Duration from_nsec(double nsec);

    // from_time() handles two types of time format
    // 'hh:mm:ss.xxx[xxxxxx]'
    // 'NNh', 'NNm', 'NNs', 'NNms' 'NNus' 'NNns' (where NN is a number)
    static bool     is_valid_time(const char *str);
    static Duration from_time(const char *str);

    // convert to another time unit
    constexpr int64_t to_sec() const;
    constexpr int64_t to_msec() const;
    constexpr int64_t to_usec() const;
    constexpr int64_t to_nsec() const;
    constexpr double  to_double() const;  // Floating point seconds

    // Convert to a string format
    // todo: will give bad result if more than 1d!
    std::string   to_string() const;
    std::ostream &to_ostream(std::ostream &output) const;

    // check time duration validity
    // (true if time is within non-inclusive min_duration/max_duration range)
    explicit constexpr operator bool() const;
    constexpr bool     is_greater_than_zero() const;

    // comparison operators
    constexpr bool operator==(const Duration &rhs) const;
    constexpr bool operator!=(const Duration &rhs) const;
    constexpr bool operator<(const Duration &rhs) const;
    constexpr bool operator<=(const Duration &rhs) const;
    constexpr bool operator>(const Duration &rhs) const;
    constexpr bool operator>=(const Duration &rhs) const;

    // mathematical operators (const)
    constexpr Duration operator+() const;
    constexpr Duration operator-() const;

    constexpr Duration operator+(const Duration &rhs) const;
    constexpr Duration operator-(const Duration &rhs) const;

    constexpr Duration operator*(int64_t factor) const;
    constexpr Duration operator/(int64_t factor) const;

    // mathematical operators (modifying)
    Duration &operator+=(const Duration &rhs);
    Duration &operator-=(const Duration &rhs);

    Duration &operator*=(int64_t factor);
    Duration &operator/=(int64_t factor);

    // mathematical Functions
    static constexpr Duration abs(const Duration &duration);

    Duration &abs();

  private:
    explicit constexpr Duration(int64_t nsec);

    int64_t nsec_;  // nanoseconds
};

constexpr Duration::Duration() : nsec_(0)
{
}

constexpr Duration::Duration(int64_t nsec)  // private!
    : nsec_(nsec)
{
}

constexpr Duration Duration::zero()
{
    return Duration(0);
}

constexpr Duration Duration::one()
{
    return Duration(1);
}

constexpr Duration Duration::min_duration()
{
    // arbitrary number exceeding -100 years (-4500000000000000000 ns)
    return Duration(-4500000000000000000L);
}

constexpr Duration Duration::max_duration()
{
    // arbitrary number exceeding +100 years (+4500000000000000000 ns)
    return Duration(+4500000000000000000L);
}

constexpr Duration Duration::from_hour(int64_t hour)
{
    return Duration(hour * 3600L * std::nano::den);
}

constexpr Duration Duration::from_min(int64_t min)
{
    return Duration(min * 60L * std::nano::den);
}

constexpr Duration Duration::from_sec(int64_t sec)
{
    return Duration(sec * std::nano::den);
}

constexpr Duration Duration::from_msec(int64_t msec)
{
    return Duration(msec * std::ratio_divide<std::nano, std::milli>::den);
}

constexpr Duration Duration::from_usec(int64_t usec)
{
    return Duration(usec * std::ratio_divide<std::nano, std::micro>::den);
}

constexpr Duration Duration::from_nsec(int64_t nsec)
{
    return Duration(nsec);
}

constexpr Duration Duration::from_hour(double hour)
{
    return Duration(static_cast<int64_t>(hour * 3600.0 * std::nano::den));
}

constexpr Duration Duration::from_min(double min)
{
    return Duration(static_cast<int64_t>(min * 60.0 * std::nano::den));
}

constexpr Duration Duration::from_sec(double sec)
{
    return Duration(static_cast<int64_t>(sec * std::nano::den));
}

constexpr Duration Duration::from_msec(double msec)
{
    return Duration(static_cast<int64_t>(msec * std::ratio_divide<std::nano, std::milli>::den));
}

constexpr Duration Duration::from_usec(double usec)
{
    return Duration(static_cast<int64_t>(usec * std::ratio_divide<std::nano, std::micro>::den));
}

constexpr Duration Duration::from_nsec(double nsec)
{
    return Duration(static_cast<int64_t>(nsec));
}

constexpr int64_t Duration::to_sec() const
{
    return nsec_ / std::nano::den;
}

constexpr int64_t Duration::to_msec() const
{
    return nsec_ / std::ratio_divide<std::nano, std::milli>::den;
}

constexpr int64_t Duration::to_usec() const
{
    return nsec_ / std::ratio_divide<std::nano, std::micro>::den;
}

constexpr int64_t Duration::to_nsec() const
{
    return nsec_;
}

constexpr double Duration::to_double() const
{
    return (double)nsec_ / (double)std::nano::den;
}

inline std::string Duration::to_string() const
{
    std::ostringstream output;

    int64_t nsecs;

    if (nsec_ >= 0)
    {
        nsecs = nsec_;
    }
    else
    {
        nsecs = -nsec_;
    }

    int64_t days;
    int64_t hours;
    int64_t mins;
    int64_t secs;

    days = nsecs / (24 * 3600 * std::nano::den);
    nsecs -= days * (24 * 3600 * std::nano::den);
    hours = nsecs / (3600 * std::nano::den);
    nsecs -= hours * (3600 * std::nano::den);
    mins = nsecs / (60 * std::nano::den);
    nsecs -= mins * (60 * std::nano::den);
    secs = nsecs / std::nano::den;
    nsecs -= secs * std::nano::den;

    // if(is_negative)
    // 	output << "-";
    // else
    // 	output << "+";

    // output  << ToString( (days), ToString::Width(4, '_') )
    // 		<< 'd' << ' '
    output << ToString(hours, ToString::Width(2)) << ':' << ToString(mins, ToString::Width(2))
           << ':' << ToString(secs, ToString::Width(2)) << '.'
           << ToString(nsecs, ToString::Width(9));

    return output.str();
}

inline std::ostream &Duration::to_ostream(std::ostream &output) const
{
    int64_t nsecs;

    if (nsec_ >= 0)
    {
        nsecs = nsec_;
    }
    else
    {
        nsecs = -nsec_;
    }

    int64_t days;
    int64_t hours;
    int64_t mins;
    int64_t secs;

    days = nsecs / (24 * 3600 * std::nano::den);
    nsecs -= days * (24 * 3600 * std::nano::den);
    hours = nsecs / (3600 * std::nano::den);
    nsecs -= hours * (3600 * std::nano::den);
    mins = nsecs / (60 * std::nano::den);
    nsecs -= mins * (60 * std::nano::den);
    secs = nsecs / std::nano::den;
    nsecs -= secs * std::nano::den;

    // if(is_negative)
    // 	output << "-";
    // else
    // 	output << "+";

    // output  << ToString( (days), ToString::Width(4, '_') )
    // 		<< 'd' << ' '
    output << ToString(hours, ToString::Width(2)) << ':' << ToString(mins, ToString::Width(2))
           << ':' << ToString(secs, ToString::Width(2)) << '.'
           << ToString(nsecs, ToString::Width(9));

    return output;
}

constexpr Duration::operator bool() const
{
    return (*this > min_duration()) && (*this < max_duration());
}

constexpr bool Duration::is_greater_than_zero() const
{
    return nsec_ > 0;
}

constexpr bool Duration::operator==(const Duration &rhs) const
{
    return nsec_ == rhs.nsec_;
}

constexpr bool Duration::operator!=(const Duration &rhs) const
{
    return nsec_ != rhs.nsec_;
}

constexpr bool Duration::operator<(const Duration &rhs) const
{
    return nsec_ < rhs.nsec_;
}

constexpr bool Duration::operator<=(const Duration &rhs) const
{
    return nsec_ <= rhs.nsec_;
}

constexpr bool Duration::operator>(const Duration &rhs) const
{
    return nsec_ > rhs.nsec_;
}

constexpr bool Duration::operator>=(const Duration &rhs) const
{
    return nsec_ >= rhs.nsec_;
}

constexpr Duration Duration::operator+() const
{
    return *this;
}

constexpr Duration Duration::operator-() const
{
    return Duration(-nsec_);
}

constexpr Duration Duration::operator+(const Duration &duration) const
{
    return Duration(nsec_ + duration.nsec_);
}

constexpr Duration Duration::operator-(const Duration &duration) const
{
    return Duration(nsec_ - duration.nsec_);
}

constexpr Duration Duration::operator*(int64_t factor) const
{
    return Duration(nsec_ * factor);
}

constexpr Duration Duration::operator/(int64_t factor) const
{
    return Duration(nsec_ / factor);
}

inline Duration &Duration::operator+=(const Duration &duration)
{
    nsec_ += duration.nsec_;
    return *this;
}

inline Duration &Duration::operator-=(const Duration &duration)
{
    nsec_ -= duration.nsec_;
    return *this;
}

inline Duration &Duration::operator*=(int64_t factor)
{
    nsec_ *= factor;
    return *this;
}

inline Duration &Duration::operator/=(int64_t factor)
{
    nsec_ /= factor;
    return *this;
}

constexpr Duration Duration::abs(const Duration &duration)
{
    return duration.is_greater_than_zero() ? duration : -duration;
}

inline Duration &Duration::abs()
{
    return (*this = Duration::abs(*this));
}

inline std::ostream &operator<<(std::ostream &output, Duration duration)
{
    return duration.to_ostream(output);
}

}  // namespace alphaone

#endif
