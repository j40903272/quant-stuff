#ifndef _TAIFEXORDERMANAGERBASE_H
#define _TAIFEXORDERMANAGERBASE_H

#include "infrastructure/base/OrderManager.h"
#include "infrastructure/base/OrderReportSource.h"
#include "infrastructure/base/TaifexOrderReportListener.h"


#include <vector>

namespace alphaone
{

enum class TAIFEX_ORDER_TIMEINFORCE : unsigned char
{
    ROD = 0,
    IOC = 3,
    FOK = 4
};

enum class TAIFEX_ORDER_POSITIONEFFECT : char
{
    OPEN     = 'O',
    CLOSE    = 'C',
    DAYTRADE = 'D',
    QUOTE    = '9'
};

enum class TAIFEX_ORDER_SIDE : unsigned char
{
    BUY  = 1,
    SELL = 2
};

enum class TAIFEX_ORDERSESSION_TYPE : int
{
    FUTURE = 0,
    OPTION = 1
};

struct TaifexOrderStatus
{
    char OrderNo[5];
};

class TaifexOrderManagerBase : public OrderManager,
                               public OrderReportSource<TaifexOrderReportListener>
{
  public:
    TaifexOrderManagerBase(ObjectManager *object_manager) : OrderReportSource(object_manager)
    {
    }

    virtual ~TaifexOrderManagerBase() = default;

    virtual TaifexOrderStatus *
    NewFutureOrder(const char *pid, int price, int qty, TAIFEX_ORDER_SIDE side,
                   TAIFEX_ORDER_TIMEINFORCE timeInForce, TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                   int sessionIndex, int subSessionIndex, unsigned int account, char accountFlag,
                   const char *pUserDefine = nullptr) = 0;

    virtual void CancelFutureOrder(const char *orderno, const char *pid, TAIFEX_ORDER_SIDE side,
                                   int sessionIndex, int subSessionIndex,
                                   const char *pUserDefine = nullptr) = 0;

    virtual void Process(const Timestamp &event_loop_time) = 0;
};
}  // namespace alphaone
#endif
