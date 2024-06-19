#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include "infrastructure/common/datetime/Date.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <string>

namespace alphaone
{
struct dynamic_compare
{
    enum compare_type
    {
        less,
        greater
    };
    explicit dynamic_compare() : is_less{true}
    {
    }
    explicit dynamic_compare(const compare_type &t) : is_less{t == less}
    {
    }
    template <class T, class U>
    bool operator()(const T &t, const U &u) const
    {
        return (t != u) && ((t < u) == is_less);
    }
    bool is_less;
};
typedef size_t   AlphaFitId;
typedef uint64_t InternalOrderId;  // internal representation of order id (always starts at 1 for
                                   // actual orders ; 0 is reserved as 'invalid')
typedef uint64_t ExternalOrderId;  // external order id

typedef double BookPrice;  // price is multiplied by adjustment
typedef double BookQty;    // size is multiplied by adjustment

typedef int OrderNo;

typedef uint32_t BookNord;
typedef uint32_t BookNolv;
typedef bool     BookSide;

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
static constexpr double EPS = 1e-10;

static constexpr bool BID{true};   // either passive (at market bid) or aggressive (at market ask)
static constexpr bool ASK{false};  // either passive (at market ask) or aggressive (at market bid)

static constexpr double EMPTY_BID = 0;
static constexpr double EMPTY_ASK = 0;

static constexpr BookPrice INVALID_BID_PRICE{std::numeric_limits<double>::lowest()};
static constexpr BookPrice INVALID_ASK_PRICE{std::numeric_limits<double>::max()};

inline bool IsEqual(const double a, const double b)
{
    return std::fabs(a - b) < EPS;
}

inline bool IsZero(const double a)
{
    return std::fabs(a) < EPS;
}

template <class T>
inline int sign(T x)
{
    return IsZero(x) ? 0 : (x < 0 ? -1 : 1);
}

enum BookTrade
{
    BookPart      = 0,
    TradePart     = 1,
    BookTradePart = 2,
};

enum Side
{
    Ask    = 0,
    Bid    = 1,
    AskBid = 2,
};

enum class ProductType : int
{
    None     = 0,
    Future   = 1,
    Option   = 2,
    Security = 3,
    Warrant  = 4,
    Perp     = 5,
    Spot     = 6,
};

enum class WarrantType : int
{
    Call      = 0,
    Put       = 1,
    Bull      = 2,
    Bear      = 3,
    Undefined = 999,
};

enum CandleType : unsigned char
{
    OPEN   = 0,
    HIGH   = 1,
    LOW    = 2,
    CLOSE  = 3,
    VOLUME = 4,
    CANDLE_END,
};

enum class SubType : int
{
    Index                       = 1,
    InterestRate                = 2,
    Bond                        = 3,
    Commodity                   = 4,
    Stock                       = 5,
    Normal                      = 201,
    Leverage                    = 202,
    ReverseLeverage             = 203,
    ETF                         = 204,
    LOF                         = 205,
    ExchangeRateLeverage        = 206,
    ExchangeRateReverseLeverage = 207,
    ExchangeRateETF             = 208,
    Call                        = 401,
    Put                         = 402,
    TAS                         = 901,
    Undefined                   = 999,
};

const std::string FromProductTypeToString(ProductType type);
ProductType       FromStringToProductType(const std::string type);

enum class ENUM_CPFlag : int
{
    NONE = 0,
    CALL = 1,
    PUT  = 2
};

enum class DataSourceType : unsigned int
{
    Invalid       = 0,
    MarketByPrice = 1,
    MarketByOrder = 2,
    OptionParity  = 3,
    TPrice        = 4,
    END
};

const std::string FromDataSourceTypeToString(DataSourceType type);
DataSourceType    FromStringToDataSourceType(const std::string type);

enum class DataSourceID : unsigned char
{
    UNKNOWN             = 0,
    TAIFEX_FUTURE       = 1,
    TAIFEX_OPTION       = 2,
    TSE                 = 3,
    OTC                 = 4,
    SGX_FUTURE          = 5,
    CME                 = 6,
    TPRICE              = 7,
    SGX_REALTIME_FUTURE = 8,
    TWSE_DATA_FILE      = 9,
    BINANCE_PERP        = 10,
    END
};

enum class ProviderID : unsigned char
{
    Invalid             = 0,
    TAIFEX_FUTURE       = 1,
    TAIFEX_OPTION       = 2,
    TSE                 = 3,
    OTC                 = 4,
    SGX_FUTURE          = 5,
    CME                 = 6,
    TPRICE              = 7,
    SGX_REALTIME_FUTURE = 8,
    TWSE_DATA_FILE      = 9,
    BINANCE_PERP        = 10,
    AlphaOne            = 11,
};

enum class EngineEventLoopType : uint8_t
{
    Invalid    = 0,
    Simulation = 1,
    Production = 2,
};

enum class MarketCode : uint8_t
{
    Invalid = 0,
    Day     = 1,
    Night   = 2,
};

#pragma pack(1)
struct MarketDataFileHeader
{
    unsigned char DataSourceID;
    int64_t       ProviderTime;
    unsigned int  DataLength;
};

struct MarketDataFileStruct
{
    MarketDataFileHeader Header;
    char                 Data[65535];
};
#pragma pack()

}  // namespace alphaone
#endif
