#ifndef _TAIFEX_ORDERMANGER_H
#define _TAIFEX_ORDERMANGER_H

#include "infrastructure/base/OrderReportSource.h"
#include "infrastructure/base/TaifexOrderManagerBase.h"
#include "infrastructure/base/TaifexOrderReportListener.h"
#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/platform/manager/ObjectManager.h"
#include "infrastructure/common/order/taifex/TaifexOrderSession.h"

#include <string>
#include <vector>

namespace alphaone
{

class TaifexOrderManager : public TaifexOrderManagerBase
{
  public:
    TaifexOrderManager(ObjectManager *object_manager);
    ~TaifexOrderManager();

    TaifexOrderStatus *NewFutureOrder(const char *pid, int price, int qty, TAIFEX_ORDER_SIDE side,
                                      TAIFEX_ORDER_TIMEINFORCE    timeInForce,
                                      TAIFEX_ORDER_POSITIONEFFECT positionEffect, int sessionIndex,
                                      int subSessionIndex, unsigned int account, char accountFlag,
                                      const char *pUserDefine = nullptr);

    void CancelFutureOrder(const char *orderno, const char *pid, TAIFEX_ORDER_SIDE side,
                           int sessionIndex, int subSessionIndex,
                           const char *pUserDefine = nullptr);

    TaifexOrderStatus *NewFutureDoubleOrder(const char *pid, int bidprice, int bidqty, int askprice,
                                            int askqty, TAIFEX_ORDER_TIMEINFORCE timeInForce,
                                            TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                                            int sessionIndex, int subSessionIndex,
                                            unsigned int account, char accountFlag,
                                            const char *pUserDefine = nullptr);

    void CancelFutureDoubleOrder(const char *orderno, const char *pid, int sessionIndex,
                                 int subSessionIndex, const char *pUserDefine = nullptr);


    void Process(const Timestamp &event_loop_time);

  private:
    ObjectManager *object_manager_;

    std::vector<TaifexOrderSession *> taifex_future_order_session_list_;
    std::vector<TaifexOrderStatus>    future_orderstatus_list_;

    int next_future_order_index_;

    void LoadSession();

    void InitOrderStatus(std::string begin, std::string end,
                         std::vector<TaifexOrderStatus> &orderstatus_list);
    void ProcessReport(TaifexOrderSession *order_session);
    bool CheckTaifexAccount(const char *pUserDefine, unsigned int account, char accountFlag);
};
}  // namespace alphaone
#endif
