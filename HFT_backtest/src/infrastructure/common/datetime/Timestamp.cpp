#include "Timestamp.h"

#include "infrastructure/common/util/Branch.h"
#include "infrastructure/common/util/ToString.h"

#include <cstring>  // strerror, strlen
#include <ctime>    // ::clock_gettime
#include <iostream>

namespace alphaone
{
Timestamp Timestamp::midnight()
{
    struct timespec now_timespec = Timestamp::now().to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return epoch();
    }

    // Use Timestamp::from_date to determine midnight
    return from_date(
        Date::from_year_month_day(broken.tm_year + 1900, broken.tm_mon + 1, broken.tm_mday));
}

Timestamp Timestamp::from_date(const Date &date)
{
    // Create a broken down time structure from the input date

    struct tm broken;

    broken.tm_sec   = 0;
    broken.tm_min   = 0;
    broken.tm_hour  = 0;
    broken.tm_mday  = date.day();
    broken.tm_mon   = date.month() - 1;
    broken.tm_year  = date.year() - 1900;
    broken.tm_wday  = -1;
    broken.tm_yday  = -1;
    broken.tm_isdst = -1;

    // Use local time to get the epoch time

    time_t epoch_sec = ::mktime(&broken);

    if (epoch_sec == -1)
    {
        std::cerr << "ERROR: Bad return value from mktime (errno=" << ::strerror(errno) << ")"
                  << '\n';
        abort();
    }

    return Timestamp::from_epoch_sec(static_cast<int64_t>(epoch_sec));
}

Timestamp Timestamp::from_time(const char *s)
{
    std::size_t len = ::strlen(s);
    // int16_t yyyy = (s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0');
    // int8_t mm = (s[5] - '0') * 10 + (s[6] - '0');
    // int8_t dd = (s[8] - '0') * 10 + (s[9] - '0');
    int8_t  hh = (s[0] - '0') * 10 + (s[1] - '0');
    int8_t  nn = (s[3] - '0') * 10 + (s[4] - '0');
    int8_t  ss = (s[6] - '0') * 10 + (s[7] - '0');
    int64_t ns = 0;
    if (len >= 10)
        ns += (s[9] - '0') * 100000000;
    if (len >= 11)
        ns += (s[10] - '0') * 10000000;
    if (len >= 12)
        ns += (s[11] - '0') * 1000000;
    if (len >= 13)
        ns += (s[12] - '0') * 100000;
    if (len >= 14)
        ns += (s[13] - '0') * 10000;
    if (len >= 15)
        ns += (s[14] - '0') * 1000;
    if (len >= 16)
        ns += (s[15] - '0') * 100;
    if (len >= 17)
        ns += (s[16] - '0') * 10;
    if (len >= 18)
        ns += (s[17] - '0') * 1;

    struct tm broken;

    broken.tm_sec   = ss;
    broken.tm_min   = nn;
    broken.tm_hour  = hh;
    broken.tm_mday  = 1;
    broken.tm_mon   = 0;
    broken.tm_year  = 100;  // this is year 2000
    broken.tm_wday  = -1;
    broken.tm_yday  = -1;
    broken.tm_isdst = -1;

    // Use local time to get the epoch time

    time_t epoch_sec = ::mktime(&broken);

    if (epoch_sec == -1)
    {
        std::cerr << "ERROR: Bad return value from mktime (errno=" << ::strerror(errno) << ")"
                  << '\n';
        abort();
    }

    // std::cout << "INTERNAL s=" << s << " len=" << len << " yyyy=" << 0 << " mm=" << 0 << " dd="
    // << 0
    //		  << " hh=" << (int16_t) hh << " nn=" << (int16_t) nn << " ss=" << (int16_t) ss
    //		  << " xx=" << ns
    //		  << '\n';

    // Success!!!

    return Timestamp::from_epoch_nsec(static_cast<int64_t>(epoch_sec * 1000000000) + ns);
}

Timestamp Timestamp::from_date_time(const Date &date, const char *s)
{
    std::size_t len = ::strlen(s);
    if (BRANCH_UNLIKELY(s[2] != ':' && s[5] != ':'))
    {
        std::cerr << "Timestamp::from_date_time malformed date_time (needs hh:mm not h:mm) -- s="
                  << s << '\n';
        abort();
    }
    // int16_t yyyy = (s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0');
    // int8_t mm = (s[5] - '0') * 10 + (s[6] - '0');
    // int8_t dd = (s[8] - '0') * 10 + (s[9] - '0');
    int8_t hh = (s[0] - '0') * 10 + (s[1] - '0');
    int8_t nn = (s[3] - '0') * 10 + (s[4] - '0');
    int8_t ss = 0;
    if (len >= 7)
        ss = (s[6] - '0') * 10 + (s[7] - '0');
    int64_t ns = 0;
    if (len >= 10)
        ns += (s[9] - '0') * 100000000;
    if (len >= 11)
        ns += (s[10] - '0') * 10000000;
    if (len >= 12)
        ns += (s[11] - '0') * 1000000;
    if (len >= 13)
        ns += (s[12] - '0') * 100000;
    if (len >= 14)
        ns += (s[13] - '0') * 10000;
    if (len >= 15)
        ns += (s[14] - '0') * 1000;
    if (len >= 16)
        ns += (s[15] - '0') * 100;
    if (len >= 17)
        ns += (s[16] - '0') * 10;
    if (len >= 18)
        ns += (s[17] - '0') * 1;

    struct tm broken;

    broken.tm_sec   = ss;
    broken.tm_min   = nn;
    broken.tm_hour  = hh;
    broken.tm_mday  = date.day();
    broken.tm_mon   = date.month() - 1;
    broken.tm_year  = date.year() - 1900;
    broken.tm_wday  = -1;
    broken.tm_yday  = -1;
    broken.tm_isdst = -1;

    // std::cout << "INTERNAL s=" << s << " len=" << len << " yyyy=" << date.year() << " mm=" <<
    // date.month() << " dd=" << date.day()
    //		  << " hh=" << (int16_t) hh << " nn=" << (int16_t) nn << " ss=" << (int16_t) ss
    //		  << " xx=" << ns
    //		  << '\n';

    // Use local time to get the epoch time

    time_t epoch_sec = ::mktime(&broken);

    if (epoch_sec == -1)
    {
        std::cerr << "ERROR: Bad return value from mktime (errno=" << ::strerror(errno) << ")"
                  << '\n';
        abort();
    }

    // Success!!!

    return Timestamp::from_epoch_nsec(static_cast<int64_t>(epoch_sec * 1000000000) + ns);
}

Timestamp Timestamp::from_date_time(const char *s)
{
    std::size_t len  = ::strlen(s);
    int16_t     yyyy = (s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0');
    int8_t      mm   = (s[5] - '0') * 10 + (s[6] - '0');
    int8_t      dd   = (s[8] - '0') * 10 + (s[9] - '0');
    int8_t      hh   = (s[11] - '0') * 10 + (s[12] - '0');
    int8_t      nn   = (s[14] - '0') * 10 + (s[15] - '0');
    int8_t      ss   = (s[17] - '0') * 10 + (s[18] - '0');
    int64_t     ns   = 0;
    if (len >= 21)
        ns += (s[20] - '0') * 100000000;
    if (len >= 22)
        ns += (s[21] - '0') * 10000000;
    if (len >= 23)
        ns += (s[22] - '0') * 1000000;
    if (len >= 24)
        ns += (s[23] - '0') * 100000;
    if (len >= 25)
        ns += (s[24] - '0') * 10000;
    if (len >= 26)
        ns += (s[25] - '0') * 1000;
    if (len >= 27)
        ns += (s[26] - '0') * 100;
    if (len >= 28)
        ns += (s[27] - '0') * 10;
    if (len >= 29)
        ns += (s[28] - '0') * 1;

    struct tm broken;

    broken.tm_sec   = ss;
    broken.tm_min   = nn;
    broken.tm_hour  = hh;
    broken.tm_mday  = dd;
    broken.tm_mon   = mm - 1;
    broken.tm_year  = yyyy - 1900;
    broken.tm_wday  = -1;
    broken.tm_yday  = -1;
    broken.tm_isdst = -1;

    // std::cout << "INTERNAL s=" << s << " len=" << len << " yyyy=" << yyyy << " mm=" << (int16_t)
    // mm << " dd=" << (int16_t) dd
    //		  << " hh=" << (int16_t) hh << " nn=" << (int16_t) nn << " ss=" << (int16_t) ss
    //		  << " xx=" << ns
    //		  << '\n';

    // Use local time to get the epoch time

    time_t epoch_sec = ::mktime(&broken);

    if (epoch_sec == -1)
    {
        std::cerr << "ERROR: Bad return value from mktime (errno=" << ::strerror(errno) << ")"
                  << '\n';
        abort();
    }

    // Success!!!

    return Timestamp::from_epoch_nsec(static_cast<int64_t>(epoch_sec * 1000000000) + ns);
}

Date Timestamp::to_date() const
{
    // Get the current time (as a timespec)
    struct timespec now_timespec = to_timespec();

    // Get the broken down time
    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return Date::invalid_date();
    }

    return Date::from_year_month_day(broken.tm_year + 1900, broken.tm_mon + 1, broken.tm_mday);
}

std::string Timestamp::to_string() const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return std::string("INVALID-TIMEPOINT");
    }

    // Format as text w/nanoseconds "21:49:08.606901333"

    constexpr ToString::Width width_2(2);
    constexpr ToString::Width width_9(9);

    std::string output;

    output.reserve(20);
    output += ToString(broken.tm_hour, width_2).c_str();
    output += ':';
    output += ToString(broken.tm_min, width_2).c_str();
    output += ':';
    output += ToString(broken.tm_sec, width_2).c_str();
    output += '.';
    output += ToString(now_timespec.tv_nsec, width_9).c_str();

    return output;
}

std::string Timestamp::to_string_ctime() const
{
    // ctime_r populates a 26 byte buffer with the following format:
    // "Wed Jun 30 21:49:08 1993\n" (note byte 25 is '\0')

    const time_t epoch_sec = static_cast<time_t>(to_epoch_sec());

    char buffer[26];
    ::ctime_r(&epoch_sec, buffer);
    buffer[24] = '\0';  // Remove '\n'

    return std::string(buffer);
}

std::string Timestamp::to_string_date(char sep) const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return std::string("INVALID-TIMEPOINT");
    }

    // Format as text YYYYMMDD "20130413"

    constexpr ToString::Width width_2(2);
    constexpr ToString::Width width_4(4);

    std::string output;

    output.reserve(10);
    output += ToString(broken.tm_year + 1900, width_4).c_str();
    output += std::string(sizeof(char), sep);
    output += ToString(broken.tm_mon + 1, width_2).c_str();
    output += std::string(sizeof(char), sep);
    output += ToString(broken.tm_mday, width_2).c_str();
    return output;
}

std::string Timestamp::to_string_date_time(char sep) const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return std::string("INVALID-TIMEPOINT");
    }

    // Format as text "2013-04-13 16:46:50.000000000"

    constexpr ToString::Width width_2(2);
    constexpr ToString::Width width_4(4);
    constexpr ToString::Width width_9(9);

    std::string output;

    output.reserve(32);

    // Date

    output += ToString(broken.tm_year + 1900, width_4).c_str();
    output += '-';
    output += ToString(broken.tm_mon + 1, width_2).c_str();
    output += '-';
    output += ToString(broken.tm_mday, width_2).c_str();

    // Time

    output += sep;
    output += ToString(broken.tm_hour, width_2).c_str();
    output += ':';
    output += ToString(broken.tm_min, width_2).c_str();
    output += ':';
    output += ToString(broken.tm_sec, width_2).c_str();
    output += '.';
    output += ToString(now_timespec.tv_nsec, width_9).c_str();

    return output;
}

std::string Timestamp::to_string_yyyymmddHHMMSS() const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return std::string("INVALID-TIMEPOINT");
    }

    // Format as text "20130413164650"

    constexpr ToString::Width width_2(2);
    constexpr ToString::Width width_4(4);

    std::string output;

    output.reserve(32);

    // Date
    output += ToString(broken.tm_year + 1900, width_4).c_str();
    output += ToString(broken.tm_mon + 1, width_2).c_str();
    output += ToString(broken.tm_mday, width_2).c_str();

    // Time
    output += ToString(broken.tm_hour, width_2).c_str();
    output += ToString(broken.tm_min, width_2).c_str();
    output += ToString(broken.tm_sec, width_2).c_str();

    return output;
}

std::string Timestamp::to_string_time() const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return std::string("INVALID-TIMEPOINT");
    }

    // Format as text HHMMSS "214908"

    constexpr ToString::Width width_2(2);

    std::string output;

    output.reserve(6);
    output += ToString(broken.tm_hour, width_2).c_str();
    output += ToString(broken.tm_min, width_2).c_str();
    output += ToString(broken.tm_sec, width_2).c_str();

    return output;
}

std::string Timestamp::to_string_hour() const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return std::string("INVALID-TIMEPOINT");
    }

    // Format as text HHMMSS "214908"

    constexpr ToString::Width width_2(2);

    std::string output;

    output.reserve(6);
    output += ToString(broken.tm_hour, width_2).c_str();

    return output;
}

std::ostream &Timestamp::to_ostream(std::ostream &output) const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return (output << "INVALID-TIMEPOINT");
    }

    // Format as text w/nanoseconds "21:49:08.606901333"

    constexpr ToString::Width width_2(2);
    constexpr ToString::Width width_9(9);

    output << ToString(broken.tm_hour, width_2);
    output << ':';
    output << ToString(broken.tm_min, width_2);
    output << ':';
    output << ToString(broken.tm_sec, width_2);
    output << '.';
    output << ToString(now_timespec.tv_nsec, width_9);

    return output;
}

std::ostream &Timestamp::to_ostream_ctime(std::ostream &output) const
{
    // ctime_r populates a 26 byte buffer with the following format:
    // "Wed Jun 30 21:49:08 1993\n" (note byte 25 is '\0')

    const time_t epoch_sec = static_cast<time_t>(to_epoch_sec());

    char buffer[26];
    ::ctime_r(&epoch_sec, buffer);
    buffer[24] = '\0';  // Remove '\n'

    return (output << buffer);
}

std::ostream &Timestamp::to_ostream_date(std::ostream &output) const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return (output << "INVALID-TIMEPOINT");
    }

    // Format as text YYYYMMDD "20130413"

    constexpr ToString::Width width_2(2);
    constexpr ToString::Width width_4(4);

    output << ToString(broken.tm_year + 1900, width_4).c_str();
    output << ToString(broken.tm_mon + 1, width_2).c_str();
    output << ToString(broken.tm_mday, width_2).c_str();

    return output;
}

std::ostream &Timestamp::to_ostream_date_time(std::ostream &output) const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return (output << "INVALID-TIMEPOINT");
    }

    // Format as text "2013-04-13 16:46:50.000000000"

    constexpr ToString::Width width_2(2);
    constexpr ToString::Width width_4(4);
    constexpr ToString::Width width_9(9);

    // Date

    output << ToString(broken.tm_year + 1900, width_4).c_str();
    output << '-';
    output << ToString(broken.tm_mon + 1, width_2).c_str();
    output << '-';
    output << ToString(broken.tm_mday, width_2).c_str();

    // Time

    output << ' ';
    output << ToString(broken.tm_hour, width_2).c_str();
    output << ':';
    output << ToString(broken.tm_min, width_2).c_str();
    output << ':';
    output << ToString(broken.tm_sec, width_2).c_str();
    output << '.';
    output << ToString(now_timespec.tv_nsec, width_9).c_str();

    return output;
}

std::ostream &Timestamp::to_ostream_time(std::ostream &output) const
{
    // Get the current time (as a timespec)

    struct timespec now_timespec = to_timespec();

    // Get the broken down time

    struct tm broken;

    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';

        return (output << "INVALID-TIMEPOINT");
    }

    // Format as text HHMMSS "214908"

    constexpr ToString::Width width_2(2);

    output << ToString(broken.tm_hour, width_2).c_str();
    output << ToString(broken.tm_min, width_2).c_str();
    output << ToString(broken.tm_sec, width_2).c_str();

    return output;
}

int64_t Timestamp::now_system_epoch_nsec()
{
    struct timespec time_now;

    if (::clock_gettime(CLOCK_REALTIME, &time_now) == 0)
    {
        return (static_cast<int64_t>(time_now.tv_sec) * std::nano::den) +
               static_cast<int64_t>(time_now.tv_nsec);
    }

    std::cerr << "Timestamp: Bad return value from clock_gettime (errno=" << ::strerror(errno)
              << ")" << '\n';

    return 0;
}

int Timestamp::year() const
{
    return get_tm().tm_year + 1900;
}

int Timestamp::month() const
{
    return get_tm().tm_mon + 1;
}

int Timestamp::day() const
{
    return get_tm().tm_mday;
}

int Timestamp::hour() const
{
    return get_tm().tm_hour;
}

int Timestamp::minute() const
{
    return get_tm().tm_min;
}

int Timestamp::second() const
{
    return get_tm().tm_sec;
}

const struct tm Timestamp::get_tm() const
{
    struct timespec now_timespec = to_timespec();
    struct tm       broken;
    if (BRANCH_UNLIKELY(::localtime_r(&now_timespec.tv_sec, &broken) == nullptr))
    {
        std::cerr << "ERROR: Bad return value from localtime_r (errno=" << ::strerror(errno) << ")"
                  << '\n';
        abort();
    }
    return broken;
}

}  // namespace alphaone
