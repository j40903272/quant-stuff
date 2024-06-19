#include "Duration.h"

#include <cctype>
#include <cstring>  // strerror, strlen
#include <iostream>

bool alphaone::Duration::is_valid_time(const char *s)
{
    std::size_t len = ::strlen(s);

    // check format for 'hh:mm:ss.xxx[xxxxxx]'
    if (len < 12 || !isdigit(s[0]) || !isdigit(s[1]) || !isdigit(s[3]) || !isdigit(s[4]) ||
        !isdigit(s[6]) || !isdigit(s[7]) || !isdigit(s[9]) || !isdigit(s[10]) || !isdigit(s[11]))
    {
        std::cout << "Malformed duration, s=" << s << '\n';
        return false;
    }

    return true;
}

alphaone::Duration alphaone::Duration::from_time(const char *s)
{
    std::size_t len = ::strlen(s);

    // check format for a number and then 'h' 'm' 's' 'ms' 'us' or 'ns'
    if (len >= 2 && isdigit(s[0]) && (s[len - 1] == 'h' || s[len - 1] == 'm' || s[len - 1] == 's'))
    {
        const double num = atof(s);
        if (s[len - 1] == 'h')
            return Duration::from_hour(num);
        else if (s[len - 1] == 'm')
            return Duration::from_min(num);
        else if (s[len - 1] == 's')
        {
            if (s[len - 2] == 'm')
                return Duration::from_msec(num);
            else if (s[len - 2] == 'u')
                return Duration::from_usec(num);
            else if (s[len - 2] == 'n')
                return Duration::from_nsec(num);
            return Duration::from_sec(num);
        }
    }

    // check format for 'hh:mm:ss.xxx[xxxxxx]'
    if (len < 12 || !isdigit(s[0]) || !isdigit(s[1]) || !isdigit(s[3]) || !isdigit(s[4]) ||
        !isdigit(s[6]) || !isdigit(s[7]) || !isdigit(s[9]) || !isdigit(s[10]) || !isdigit(s[11]) ||
        (len > 18))
    {
        std::cout << "Malformed duration, pls double check duration_string=[" << s << "]" << '\n';
        abort();
    }

    int64_t hh = (s[0] - '0') * 10 + (s[1] - '0');
    int64_t nn = (s[3] - '0') * 10 + (s[4] - '0');
    int64_t ss = (s[6] - '0') * 10 + (s[7] - '0');
    int64_t ns = (s[9] - '0') * 100000000 + (s[10] - '0') * 10000000 + (s[11] - '0') * 1000000;
    if (len >= 13 && isdigit(s[12]))
        ns += (s[12] - '0') * 100000;
    if (len >= 14 && isdigit(s[13]))
        ns += (s[13] - '0') * 10000;
    if (len >= 15 && isdigit(s[14]))
        ns += (s[14] - '0') * 1000;
    if (len >= 16 && isdigit(s[15]))
        ns += (s[15] - '0') * 100;
    if (len >= 17 && isdigit(s[16]))
        ns += (s[16] - '0') * 10;
    if (len >= 18 && isdigit(s[17]))
        ns += (s[17] - '0') * 1;

    // std::cout << " DURATION:: s=" << std::string(s) << " h=" << hh << " m=" << nn << " s=" << ss
    // << " ns=" << ns << '\n';

    return Duration::from_hour(hh) + Duration::from_min(nn) + Duration::from_sec(ss) +
           Duration::from_nsec(ns);
}
