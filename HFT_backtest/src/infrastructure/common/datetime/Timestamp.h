#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/datetime/Duration.h"

// clang-format off
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/spdlog/fmt/ostr.h"
// clang-format on

#include <cstdint>
#include <limits>
#include <ratio>  // std::nano::den

namespace alphaone
{
class Timestamp
{
  public:
    // Timestamp is a point in time stored in nanoseconds since UNIX epoch
    constexpr Timestamp();
    constexpr Timestamp(const Timestamp &) = default;
    Timestamp &operator=(const Timestamp &) = default;

    ~Timestamp() = default;

    // constants
    static constexpr Timestamp epoch();  // UNIX epoch, nsec=0 (is a valid timestamp)
    static constexpr Timestamp min_time();
    static constexpr Timestamp max_time();
    static constexpr Timestamp invalid();

    // get current time
    static Timestamp now();  // use best available

    // get the local current midnight time
    static Timestamp midnight();       // Previous midnight (today)
    static Timestamp midnight_next();  // Next midnight (tomorrow)

    // check valid
    bool                is_valid() const;
    [[deprecated]] bool __deprecated_is_valid__() const;

    // convert from another time format
    static Timestamp from_date(const Date &date);
    static Timestamp from_time(const char *str);  // handles hh:mm:ss.xxx[xxxxxx]
    static Timestamp from_date_time(const Date &date,
                                    const char *str);  // handles date, hh:mm:ss.xxx[xxxxxx]
    static Timestamp from_date_time(const char *str);  // handles yyyy-mm-dd hh:mm:ss.xxx[xxxxxx]

    static constexpr Timestamp from_epoch_sec(int32_t sec);
    static constexpr Timestamp from_epoch_msec(int32_t msec);
    static constexpr Timestamp from_epoch_usec(int32_t usec);
    static constexpr Timestamp from_epoch_nsec(int32_t nsec);

    static constexpr Timestamp from_epoch_sec(int64_t sec);
    static constexpr Timestamp from_epoch_msec(int64_t msec);
    static constexpr Timestamp from_epoch_usec(int64_t usec);
    static constexpr Timestamp from_epoch_nsec(int64_t nsec);

    static constexpr Timestamp from_epoch_sec(double sec);
    static constexpr Timestamp from_epoch_msec(double msec);
    static constexpr Timestamp from_epoch_usec(double usec);
    static constexpr Timestamp from_epoch_nsec(double nsec);

    static constexpr Timestamp from_timeval(const struct timeval &timeval);
    static constexpr Timestamp from_timespec(const struct timespec &timespec);

    // convert to another timeformat
    Date              to_date() const;
    constexpr int64_t to_epoch_day() const;
    constexpr int64_t to_epoch_min() const;
    constexpr int64_t to_epoch_sec() const;
    constexpr int64_t to_epoch_msec() const;
    constexpr int64_t to_epoch_usec() const;
    constexpr int64_t to_epoch_nsec() const;
    constexpr double  to_epoch_double() const;  // Floating point seconds
    constexpr int64_t to_min_after_midnight() const;
    constexpr int64_t to_sec_after_midnight() const;
    constexpr int64_t to_msec_after_midnight() const;
    constexpr int64_t to_usec_after_midnight() const;
    constexpr int64_t to_nsec_after_midnight() const;

    constexpr struct timeval  to_timeval() const;
    constexpr struct timespec to_timespec() const;

    // convert to a string format
    std::string to_string() const;                          // ex: "16:46:50.000000000"
    std::string to_string_ctime() const;                    // ex: "Sat Apr 13 16:46:50 2013"
    std::string to_string_date(char sep) const;             // ex: "20130413"
    std::string to_string_date_time(char sep = ' ') const;  // ex: "2013-04-13 16:46:50.000000000"
    std::string to_string_time() const;                     // ex: "164650"
    std::string to_string_hour() const;                     // ex: "16"
    std::string to_string_yyyymmddHHMMSS() const;           // ex: "20130413164650"

    std::ostream &to_ostream(std::ostream &output) const;
    std::ostream &to_ostream_ctime(std::ostream &output) const;
    std::ostream &to_ostream_date(std::ostream &output) const;
    std::ostream &to_ostream_date_time(std::ostream &output) const;
    std::ostream &to_ostream_time(std::ostream &output) const;

    // comparison operators
    constexpr bool operator==(const Timestamp &rhs) const;
    constexpr bool operator!=(const Timestamp &rhs) const;
    constexpr bool operator<(const Timestamp &rhs) const;
    constexpr bool operator<=(const Timestamp &rhs) const;
    constexpr bool operator>(const Timestamp &rhs) const;
    constexpr bool operator>=(const Timestamp &rhs) const;

    // math operators
    constexpr Timestamp operator+(const Duration &duration) const;
    constexpr Timestamp operator-(const Duration &duration) const;
    Timestamp &         operator+=(const Duration &duration);
    Timestamp &         operator-=(const Duration &duration);
    constexpr Duration  operator-(const Timestamp &timestamp) const;

    // getters
    int year() const;    // ex: 2018
    int month() const;   // 1-12
    int day() const;     // 1-31
    int hour() const;    // 0-23
    int minute() const;  // 0-59
    int second() const;  // 0-59

    template <typename OStream>
    friend OStream &operator<<(OStream &os, const Timestamp &t)
    {
        os << "[" << t.to_string_date_time() << "]";
        return os;
    }

  private:
    explicit constexpr Timestamp(int64_t nsec);

    int64_t nsec_;  // nanoseconds

    // uses clock_gettime to get time (todo: change to tsc)
    static int64_t now_system_epoch_nsec();

    // shortcut to getting tm
    const struct tm get_tm() const;
};

constexpr Timestamp::Timestamp() : nsec_(std::numeric_limits<std::int64_t>::min())
{
}

constexpr Timestamp::Timestamp(int64_t nsec) : nsec_(nsec)  // private
{
}

constexpr Timestamp Timestamp::epoch()
{
    return Timestamp(0);
}

constexpr Timestamp Timestamp::min_time()
{
    return Timestamp(-2208988800000000000L);  // Jan 1, 1900 (-2208988800000000000 ns)
}

constexpr Timestamp Timestamp::max_time()
{
    return Timestamp(+4102444800000000000L);  // Jan 1, 2100 (+4102444800000000000 ns)
}

constexpr Timestamp Timestamp::invalid()
{
    static_assert(std::numeric_limits<std::int64_t>::min() < min_time().nsec_,
                  "Timestamp invalid value is within valid range");
    return Timestamp(std::numeric_limits<std::int64_t>::min());
}

inline Timestamp Timestamp::now()
{
    return Timestamp(Timestamp::now_system_epoch_nsec());
}

inline Timestamp Timestamp::midnight_next()
{
    return midnight() + Duration::from_hour((int64_t)24);
}

inline bool Timestamp::is_valid() const
{
    return nsec_ >= min_time().to_epoch_nsec() && nsec_ <= max_time().to_epoch_nsec();
}

inline bool Timestamp::__deprecated_is_valid__() const
{
    return nsec_ >= 0;
}

constexpr Timestamp Timestamp::from_epoch_sec(int32_t sec)
{
    return Timestamp(static_cast<int64_t>(sec * std::nano::den));
}

constexpr Timestamp Timestamp::from_epoch_msec(int32_t msec)
{
    return Timestamp(static_cast<int64_t>(msec * std::ratio_divide<std::nano, std::milli>::den));
}

constexpr Timestamp Timestamp::from_epoch_usec(int32_t usec)
{
    return Timestamp(static_cast<int64_t>(usec * std::ratio_divide<std::nano, std::micro>::den));
}

constexpr Timestamp Timestamp::from_epoch_nsec(int32_t nsec)
{
    return Timestamp(static_cast<int64_t>(nsec));
}

constexpr Timestamp Timestamp::from_epoch_sec(int64_t sec)
{
    return Timestamp(sec * std::nano::den);
}

constexpr Timestamp Timestamp::from_epoch_msec(int64_t msec)
{
    return Timestamp(msec * std::ratio_divide<std::nano, std::milli>::den);
}

constexpr Timestamp Timestamp::from_epoch_usec(int64_t usec)
{
    return Timestamp(usec * std::ratio_divide<std::nano, std::micro>::den);
}

constexpr Timestamp Timestamp::from_epoch_nsec(int64_t nsec)
{
    return Timestamp(nsec);
}

constexpr Timestamp Timestamp::from_epoch_sec(double sec)
{
    return Timestamp(static_cast<int64_t>(sec * std::nano::den));
}

constexpr Timestamp Timestamp::from_epoch_msec(double msec)
{
    return Timestamp(static_cast<int64_t>(msec * std::ratio_divide<std::nano, std::milli>::den));
}

constexpr Timestamp Timestamp::from_epoch_usec(double usec)
{
    return Timestamp(static_cast<int64_t>(usec * std::ratio_divide<std::nano, std::micro>::den));
}

constexpr Timestamp Timestamp::from_epoch_nsec(double nsec)
{
    return Timestamp(static_cast<int64_t>(nsec));
}

constexpr Timestamp Timestamp::from_timeval(const struct timeval &timeval)
{
    return Timestamp(static_cast<int64_t>(timeval.tv_sec) * std::nano::den +
                     static_cast<int64_t>(timeval.tv_usec) *
                         std::ratio_divide<std::nano, std::micro>::den);
}

constexpr Timestamp Timestamp::from_timespec(const struct timespec &timespec)
{
    return Timestamp(static_cast<int64_t>(timespec.tv_sec) * std::nano::den +
                     static_cast<int64_t>(timespec.tv_nsec));
}

constexpr int64_t Timestamp::to_epoch_day() const
{
    return to_epoch_sec() / (24 * 3600);
}

constexpr int64_t Timestamp::to_epoch_min() const
{
    return nsec_ / (60 * std::nano::den);
}

constexpr int64_t Timestamp::to_epoch_sec() const
{
    return nsec_ / std::nano::den;
}

constexpr int64_t Timestamp::to_epoch_msec() const
{
    return nsec_ / std::ratio_divide<std::nano, std::milli>::den;
}

constexpr int64_t Timestamp::to_epoch_usec() const
{
    return nsec_ / std::ratio_divide<std::nano, std::micro>::den;
}

constexpr int64_t Timestamp::to_epoch_nsec() const
{
    return nsec_;
}

constexpr double Timestamp::to_epoch_double() const
{
    return static_cast<double>(nsec_) / std::nano::den;
}

constexpr int64_t Timestamp::to_min_after_midnight() const
{
    return to_epoch_min() % (24 * 60);
}

constexpr int64_t Timestamp::to_sec_after_midnight() const
{
    return to_epoch_sec() % (24 * 60 * 60);
}

constexpr int64_t Timestamp::to_msec_after_midnight() const
{
    return to_epoch_msec() % (24 * 60 * 60 * 1000L);
}

constexpr int64_t Timestamp::to_usec_after_midnight() const
{
    return to_epoch_usec() % (24 * 60 * 60 * 1000L * 1000L);
}

constexpr int64_t Timestamp::to_nsec_after_midnight() const
{
    return to_epoch_nsec() % (24 * 60 * 60 * 1000L * 1000L * 1000L);
}

constexpr struct timeval Timestamp::to_timeval() const
{
    return timeval{
        static_cast<long int>(to_epoch_sec()),
        static_cast<decltype(timeval::tv_usec)>(
            to_epoch_usec() - to_epoch_sec() * std::ratio_divide<std::nano, std::micro>::den)};
}

constexpr struct timespec Timestamp::to_timespec() const
{
    return timespec{to_epoch_sec(), static_cast<decltype(timespec::tv_nsec)>(
                                        to_epoch_nsec() - to_epoch_sec() * std::nano::den)};
}

constexpr bool Timestamp::operator==(const Timestamp &rhs) const
{
    return nsec_ == rhs.nsec_;
}

constexpr bool Timestamp::operator!=(const Timestamp &rhs) const
{
    return nsec_ != rhs.nsec_;
}

constexpr bool Timestamp::operator<(const Timestamp &rhs) const
{
    return nsec_ < rhs.nsec_;
}

constexpr bool Timestamp::operator<=(const Timestamp &rhs) const
{
    return nsec_ <= rhs.nsec_;
}

constexpr bool Timestamp::operator>(const Timestamp &rhs) const
{
    return nsec_ > rhs.nsec_;
}

constexpr bool Timestamp::operator>=(const Timestamp &rhs) const
{
    return nsec_ >= rhs.nsec_;
}

// inline std::ostream &operator<<(std::ostream &output, Timestamp time_point)
// {
//     return time_point.to_ostream_date_time(output);
// }

constexpr Timestamp Timestamp::operator+(const Duration &duration) const
{
    return Timestamp(nsec_ + duration.to_nsec());
}

constexpr Timestamp Timestamp::operator-(const Duration &duration) const
{
    return Timestamp(nsec_ - duration.to_nsec());
}

inline Timestamp &Timestamp::operator+=(const Duration &duration)
{
    nsec_ += duration.to_nsec();
    return *this;
}

inline Timestamp &Timestamp::operator-=(const Duration &duration)
{
    nsec_ -= duration.to_nsec();
    return *this;
}

constexpr Duration Timestamp::operator-(const Timestamp &timestamp) const
{
    return Duration::from_nsec(nsec_ - timestamp.to_epoch_nsec());
}

}  // namespace alphaone

#endif
