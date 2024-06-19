#include "FixParser.h"

#include "infrastructure/common/util/String.h"

#include <cfloat>
#include <climits>
#include <string.h>

namespace alphaone
{

TWSEFixParser::TWSEFixParser()
{
}

TWSEFixParser::~TWSEFixParser()
{
}

void TWSEFixParser::FastParse(const char *data, int len)
{
    fast_parse_result_.clear();
    std::string_view sv  = std::string_view(data, len);
    auto &&          vec = SplitSVPtr(sv, "\1");
    for (auto iter = vec.begin(); iter != vec.end(); ++iter)
    {
        auto &&vec_kv = SplitSVPtr(*iter, "=");
        int    tag    = atoi(vec_kv[0].data());
        if (tag > 0)
        {
            fast_parse_result_[tag] = std::move(vec_kv[1]);
        }
    }
}

std::string_view &TWSEFixParser::GetStringView(FIX_TAG tag)
{
    int nTag = (int)tag;
    return fast_parse_result_[nTag];
}

double TWSEFixParser::GetDouble(FIX_TAG tag)
{
    int nTag = (int)tag;
    if (fast_parse_result_.count(nTag) > 0)
    {
        char buf[255];
        int  len = fast_parse_result_[nTag].length();
        memcpy(buf, fast_parse_result_[nTag].data(), len);
        buf[len] = 0;
        return atof(buf);
    }
    return -DBL_MAX;
}

int TWSEFixParser::GetInt(FIX_TAG tag)
{
    int nTag = (int)tag;
    if (fast_parse_result_.count(nTag) > 0)
    {
        char buf[255];
        int  len = fast_parse_result_[nTag].length();
        memcpy(buf, fast_parse_result_[nTag].data(), len);
        buf[len] = 0;
        return atoi(buf);
    }
    return INT_MIN;
}
}  // namespace alphaone