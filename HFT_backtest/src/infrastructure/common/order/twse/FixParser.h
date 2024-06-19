#ifndef _FIXPARSER_H_
#define _FIXPARSER_H_

#include "FixDefine.h"

#include <functional>
#include <string_view>
#include <unordered_map>

namespace alphaone
{
class TWSEFixParser
{
  public:
    TWSEFixParser();
    ~TWSEFixParser();

    void              FastParse(const char *data, int len);
    std::string_view &GetStringView(FIX_TAG tag);
    double            GetDouble(FIX_TAG tag);
    int               GetInt(FIX_TAG tag);

  private:
    std::unordered_map<int, std::string_view> fast_parse_result_;
};

}  // namespace alphaone
#endif