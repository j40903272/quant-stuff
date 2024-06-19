#ifndef _INDEXMESSAGE_H_
#define _INDEXMESSAGE_H_

#include "infrastructure/common/message/MarketDataMessage.h"
#include "infrastructure/common/symbol/Symbol.h"

namespace alphaone
{

struct IndexMessage
{
    BookPrice             price_[AskBid];
    BookPrice             trade_;
    const Symbol *        symbol_;
    MarketDataMessageType type_;
};

}  // namespace alphaone

#endif