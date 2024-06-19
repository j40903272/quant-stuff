#ifndef MARKETDATALISTENER_H
#define MARKETDATALISTENER_H
#include "infrastructure/common/message/MarketDataMessage.h"
#include "infrastructure/common/symbol/Symbol.h"

#include <map>

namespace alphaone
{
class MarketDataListener
{
  public:
    MarketDataListener()                           = default;
    MarketDataListener(const MarketDataListener &) = delete;
    MarketDataListener &operator=(const MarketDataListener &) = delete;
    virtual ~MarketDataListener()                             = default;

    virtual void OnMarketDataMessage(const MarketDataMessage *mm, void *raw_packet) = 0;

  private:
};
}  // namespace alphaone
#endif
