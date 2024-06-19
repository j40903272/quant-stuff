#ifndef _PREMARKET_H_
#define _PREMARKET_H_

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/typedef/Typedefs.h"

#include <cstdint>

namespace alphaone
{

struct PreMarketMessage
{
    int64_t   ExchTime;
    int64_t   ApTime;
    char      Pid[SYMBOLID_LENGTH + 1];
    BookPrice DealPrice;
    BookQty   DealQty;
    BookSide  Side;
    BookPrice Bid[5];
    BookPrice Ask[5];
    BookQty   BidQty[5];
    BookQty   AskQty[5];
    BookQty   TotalQty;
    int32_t   MainType;
    int64_t   SeqNo;
};

}  // namespace alphaone


#endif