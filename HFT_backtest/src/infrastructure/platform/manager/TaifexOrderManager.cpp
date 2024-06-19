#include "TaifexOrderManager.h"

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/message/TAIFEX.h"
#include "infrastructure/common/order/taifex/TMP.h"
#include "infrastructure/common/util/Order.h"

#include <boost/algorithm/string.hpp>
#include <tuple>

namespace alphaone
{

TaifexOrderManager::TaifexOrderManager(ObjectManager *object_manager)
    : TaifexOrderManagerBase(object_manager)
    , next_future_order_index_(0)
{
    object_manager_ = object_manager;
    LoadSession();
}

TaifexOrderManager::~TaifexOrderManager()
{
}

void TaifexOrderManager::LoadSession()
{
    const auto session_config =
        object_manager_->GetGlobalConfiguration()->GetJson().at("OrderSource").at("Taifex");
    nlohmann::json future_list;
    std::string    fut_begin{"00000"}, fut_end{"00000"}, opt_begin{"00000"}, opt_end{"00000"};

    if (session_config.contains("Future") && session_config["Future"].contains("SessionList"))
        future_list = session_config["Future"]["SessionList"];

    if (future_list.is_array() && !future_list.empty())
    {
        for (auto &[future_k, future_v] : future_list.items())
        {
            int session_group_id        = atoi(future_v.value("Session_GroupID", "0").c_str());
            TaifexOrderSession *session = new TaifexOrderSession();
            session->Init(session_group_id, TAIFEX_ORDERSESSION_TYPE::FUTURE);
            taifex_future_order_session_list_.push_back(session);
        }
    }
    else
    {
        SPDLOG_WARN("config did not contains correct fields in OrderSource thus won't new "
                    "any Future Order Session");
    }


    if (taifex_future_order_session_list_.size() > 0)
    {
        if (session_config.contains("/Future/OrderNo/Begin"_json_pointer))
            fut_begin = session_config["Future"]["OrderNo"]["Begin"].get<std::string>();
        if (session_config.contains("/Future/OrderNo/End"_json_pointer))
            fut_end = session_config["Future"]["OrderNo"]["End"].get<std::string>();

        InitOrderStatus(fut_begin, fut_end, future_orderstatus_list_);

        int maxOrderIndex = 0;
        for (unsigned int i = 0; i < taifex_future_order_session_list_.size(); ++i)
        {
            int index =
                taifex_future_order_session_list_[i]->GetOrderStatusIndex(future_orderstatus_list_);
            if (index > maxOrderIndex)
            {
                maxOrderIndex = index;
            }
        }

        next_future_order_index_ = maxOrderIndex;
        printf("InitFutureOrderNo=%.5s\n",
               future_orderstatus_list_[next_future_order_index_].OrderNo);
    }
}

void TaifexOrderManager::InitOrderStatus(std::string begin, std::string end,
                                         std::vector<TaifexOrderStatus> &orderstatus_list)
{
    char tmpBegin[5];
    char tmpEnd[5];
    memcpy(tmpBegin, begin.c_str(), 5);
    memcpy(tmpEnd, end.c_str(), 5);

    int index = 0;
    while (1)
    {
        if (memcmp(tmpBegin, tmpEnd, 5) != 0)
        {
            orderstatus_list.push_back(TaifexOrderStatus());
            memcpy(&orderstatus_list[index].OrderNo[0], tmpBegin, 5);
            OrderNo_NextFn(tmpBegin);
            index++;
        }
        else
        {
            orderstatus_list.push_back(TaifexOrderStatus());
            memcpy(&orderstatus_list[index].OrderNo[0], tmpBegin, 5);
            break;
        }
    }
}

TaifexOrderStatus *TaifexOrderManager::NewFutureOrder(const char *pid, int price, int qty,
                                                      TAIFEX_ORDER_SIDE           side,
                                                      TAIFEX_ORDER_TIMEINFORCE    timeInForce,
                                                      TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                                                      int sessionIndex, int subSessionIndex,
                                                      unsigned int account, char accountFlag,
                                                      const char *pUserDefine)
{
    TaifexOrderStatus *order_status = &future_orderstatus_list_[next_future_order_index_];

    if (!CheckTaifexAccount(pUserDefine, account, accountFlag))
    {
        SPDLOG_ERROR("Wrong userdefine = {} format with account = {} and accountFlag = {}",
                     pUserDefine, account, accountFlag);
        return order_status;
    }

    taifex_future_order_session_list_[sessionIndex]->NewOrder(
        pid, price, qty, side, timeInForce, positionEffect, subSessionIndex, order_status, account,
        accountFlag, pUserDefine);
    SaveOrderMapping(order_status->OrderNo, pUserDefine, PacketLogType::TAIFEX,
                     PacketLogExecType::NEW);
    next_future_order_index_++;
    return order_status;
}

void TaifexOrderManager::CancelFutureOrder(const char *orderno, const char *pid,
                                           TAIFEX_ORDER_SIDE side, int sessionIndex,
                                           int subSessionIndex, const char *pUserDefine)
{
    taifex_future_order_session_list_[sessionIndex]->CancelOrder(orderno, pid, side,
                                                                 subSessionIndex, pUserDefine);
    SaveOrderMapping(const_cast<char *>(orderno), pUserDefine, PacketLogType::TAIFEX,
                     PacketLogExecType::CANCEL);
}

TaifexOrderStatus *TaifexOrderManager::NewFutureDoubleOrder(
    const char *pid, int bidprice, int bidqty, int askprice, int askqty,
    TAIFEX_ORDER_TIMEINFORCE timeInForce, TAIFEX_ORDER_POSITIONEFFECT positionEffect,
    int sessionIndex, int subSessionIndex, unsigned int account, char accountFlag,
    const char *pUserDefine)
{
    TaifexOrderStatus *order_status = &future_orderstatus_list_[next_future_order_index_];

    if (!CheckTaifexAccount(pUserDefine, account, accountFlag))
    {
        return order_status;
    }

    taifex_future_order_session_list_[sessionIndex]->NewDoubleOrder(
        pid, bidprice, bidqty, askprice, askqty, timeInForce, positionEffect, subSessionIndex,
        order_status, account, accountFlag, pUserDefine);
    SaveOrderMapping(order_status->OrderNo, pUserDefine, PacketLogType::TAIFEX,
                     PacketLogExecType::NEW);
    next_future_order_index_++;
    return order_status;
}

void TaifexOrderManager::CancelFutureDoubleOrder(const char *orderno, const char *pid,
                                                 int sessionIndex, int subSessionIndex,
                                                 const char *pUserDefine)
{
    taifex_future_order_session_list_[sessionIndex]->CancelDoubleOrder(
        orderno, pid, subSessionIndex, pUserDefine);
    SaveOrderMapping(const_cast<char *>(orderno), pUserDefine, PacketLogType::TAIFEX,
                     PacketLogExecType::CANCEL);
}

void TaifexOrderManager::Process(const Timestamp &event_loop_time)
{
}


bool TaifexOrderManager::CheckTaifexAccount(const char *pUserDefine, unsigned int account,
                                            char accountFlag)
{
    if (pUserDefine[0] == '2')
    {
        if (account == 8000013 && accountFlag == 'H')
        {
            return true;
        }
    }
    else if (pUserDefine[0] == '3')
    {
        if (account == 8000000 && accountFlag == 'H')
        {
            return true;
        }
    }
    else if (account == 0 && accountFlag == '2')
    {
        return true;
    }
    else if (account == 8888811 && accountFlag == '8')
    {
        return true;
    }
    return false;
}
}  // namespace alphaone
