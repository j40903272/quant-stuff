#ifndef _ENUMTOSTRING_H_
#define _ENUMTOSTRING_H_

#include "infrastructure/common/typedef/Typedefs.h"

namespace alphaone
{

class EnumToString
{
  public:
    EnumToString(/* args */) = delete;
    ~EnumToString()          = delete;
    static std::string ToString(const BookTrade &book_trade);
    static std::string ToString(const Side &side);
    static std::string ToString(const ProductType &product_type);
    static std::string ToString(const ENUM_CPFlag &cpflag);
    static std::string ToString(const DataSourceType &data_source_type);
    static std::string ToString(const DataSourceID &data_source_id);
    static std::string ToString(const ProviderID &provider_id);
    static std::string ToString(const EngineEventLoopType &event_loop_type);
};
}  // namespace alphaone

#endif
