#ifndef _TAIFEXORDERBOOK_H_
#define _TAIFEXORDERBOOK_H_
#include "infrastructure/common/message/MarketDataMessage.h"
#include "infrastructure/common/message/TAIFEX.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/util/UnPackBCD.h"
#include "unordered_map"

#include <limits>
#include <string>

namespace twone
{

enum class TaifexUpdateFlag : uint8_t
{
    Invalid     = 0,
    OrderBook   = 1,
    ImpliedBook = 2,
    NoChange    = 4
};


inline TaifexUpdateFlag operator|(TaifexUpdateFlag a, TaifexUpdateFlag b)
{
    return static_cast<TaifexUpdateFlag>(static_cast<int>(a) | static_cast<int>(b));
}

inline TaifexUpdateFlag operator&(TaifexUpdateFlag a, TaifexUpdateFlag b)
{
    return static_cast<TaifexUpdateFlag>(static_cast<int>(a) & static_cast<int>(b));
}

inline TaifexUpdateFlag &operator|=(TaifexUpdateFlag &a, TaifexUpdateFlag b)
{
    return a = a | b;
}

class TaifexOrderBook
{
  public:
    TaifexOrderBook(const alphaone::Symbol *symbol);
    ~TaifexOrderBook();

    TaifexOrderBook(const TaifexOrderBook &another);
    TaifexOrderBook &operator=(const TaifexOrderBook &another);

    alphaone::MarketDataMessage msg;

    TaifexUpdateFlag ParseI081(alphaone::TXMarketDataI081_t *pI081, bool isrewind,
                               alphaone::Timestamp &ts);

    TaifexUpdateFlag ParseI083(alphaone::TXMarketDataI083_t *pI083, bool isrewind,
                               alphaone::Timestamp &ts);

    void ParseI084(alphaone::TXMarketDataI084_O_Info_t *pI084, int lastProdSeq);
    void ParseI024(alphaone::TXMarketDataI024_t *pI024, bool isrewind, alphaone::Timestamp &ts);
    void ParseI025(alphaone::TXMarketDataI025_t *pI025, bool isrewind, alphaone::Timestamp &ts);
    void Reset();

    void Dump();

  private:
    int                     last_prod_seq_;
    const alphaone::Symbol *symbol_;

    std::unordered_map<int, void *> map_packet_;

    void AddBid(int level, int price, int qty);
    void AddAsk(int level, int price, int qty);

    void DeleteBid(int level);
    void DeleteAsk(int level);

    void ProcessRewind(alphaone::Timestamp &ts);
};
}  // namespace twone
#endif
