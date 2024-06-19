#include "Date.h"

#include "infrastructure/common/datetime/Timestamp.h"

#include <boost/date_time/gregorian/greg_date.hpp>
#include <cstring>

namespace alphaone
{
Date Date::today()
{
    return Timestamp::now().to_date();
}

Date Date::from_date_str(const char *str)
{
    // ASSUMPTION: Input string is eight digits like "YYYYMMDD"

    if (str && ::strlen(str) == 8)
    {
        return Date((str[0] - '0') * 1000 + (str[1] - '0') * 100 + (str[2] - '0') * 10 +
                        (str[3] - '0'),
                    (str[4] - '0') * 10 + (str[5] - '0'), (str[6] - '0') * 10 + (str[7] - '0'));
    }
    else
    {
        return Date::invalid_date();
    }
}

Date Date::from_gregorian_date(const boost::gregorian::date &date)
{
    const auto ymd = date.year_month_day();
    return Date::from_year_month_day((int16_t)ymd.year, (int16_t)ymd.month, (int16_t)ymd.day);
}

void Date::to_gregorian_date(boost::gregorian::date &date) const
{
    date = boost::gregorian::date(year(), month(), day());
}

std::map<int32_t, std::string> Date::to_month_name_{
    {1, "January"},   {2, "February"}, {3, "March"},     {4, "April"},
    {5, "May"},       {6, "June"},     {7, "July"},      {8, "August"},
    {9, "September"}, {10, "October"}, {11, "November"}, {12, "December"}};

std::string Date::to_month_name() const
{
    return to_month_name_[month()];
}
}  // namespace alphaone
