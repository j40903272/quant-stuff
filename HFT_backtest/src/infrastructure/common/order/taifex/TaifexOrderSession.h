#ifndef _TAIFEXORDERSESSION_H
#define _TAIFEXORDERSESSION_H

#include "infrastructure/base/TaifexOrderManagerBase.h"
#include "infrastructure/common/twone/def/Def.h"
#include "infrastructure/common/twone/ringbuffer/RingBuffer.h"
#include "infrastructure/common/util/Order.h"

#include <string>
namespace alphaone
{
class TaifexOrderSession
{
  public:
    TaifexOrderSession();
    ~TaifexOrderSession();

    void Init(int session_id, TAIFEX_ORDERSESSION_TYPE type);

    void NewOrder(const char *pid, int price, int qty, TAIFEX_ORDER_SIDE side,
                  TAIFEX_ORDER_TIMEINFORCE timeInForce, TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                  int subSessionIndex, TaifexOrderStatus *order_status, unsigned int account,
                  char accountFlag, const char *pUserDefine = nullptr);

    void CancelOrder(const char *orderno, const char *pid, TAIFEX_ORDER_SIDE side,
                     int subSessionIndex, const char *pUserDefine = nullptr);

    void NewDoubleOrder(const char *pid, int bidprice, int bidqty, int askprice, int askqty,
                        TAIFEX_ORDER_TIMEINFORCE    timeInForce,
                        TAIFEX_ORDER_POSITIONEFFECT positionEffect, int subSessionIndex,
                        TaifexOrderStatus *order_status, unsigned int account, char accountFlag,
                        const char *pUserDefine = nullptr);

    void ModifyDoubleOrder(const char *orderno, const char *pid, int bidprice, int askprice,
                           int sessionIndex, int subSessionIndex,
                           const char *pUserDefine = nullptr);

    void CancelDoubleOrder(const char *orderno, const char *pid, int subSessionIndex,
                           const char *pUserDefine = nullptr);

    int   GetSessionID();
    int   GetOrderStatusIndex(std::vector<TaifexOrderStatus> &orderstatus_list);
    void *Process();

  private:
    int                      session_id_;
    TAIFEX_ORDERSESSION_TYPE type_;

    twone::RingBuffer r01_;
    twone::RingBuffer r09_;
    twone::RingBuffer r02_;
    twone::RingBuffer r03_;

    bool CheckTaifexAccount(const char *pUserDefine, unsigned int account, char accountFlag);
};
}  // namespace alphaone
#endif
