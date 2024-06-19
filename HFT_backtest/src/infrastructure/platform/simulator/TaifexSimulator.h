#ifndef _TAIFEXSIMULATOR_H
#define _TAIFEXSIMULATOR_H

#include "infrastructure/base/BookDataListener.h"
#include "infrastructure/base/OrderReportMessage.h"
#include "infrastructure/base/OrderReportSource.h"
#include "infrastructure/base/TaifexOrderManagerBase.h"
#include "infrastructure/base/TaifexOrderReportListener.h"
#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/common/datetime/Duration.h"
#include "infrastructure/common/memory/BoostPool.h"
#include "infrastructure/common/side/Side.h"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/Order.h"
#include "infrastructure/platform/engine/Engine.h"
#include "infrastructure/platform/manager/MultiBookManager.h"
#include "infrastructure/platform/manager/ObjectManager.h"
#include "infrastructure/platform/profiler/Profiler.h"
#include "infrastructure/platform/reader/DelayReader.h"

#include <algorithm>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace alphaone
{

class TaifexSimulationOrderManager;

using BookCache = std::array<std::vector<std::pair<int, int>> *, 2>;

typedef int TaifexPrice;
typedef int TaifexQty;

enum OrderExecutionType
{
    Passive           = 0,
    Aggressive        = 1,
    PassiveAggressive = 2,
};

enum TaifexProductType
{
    Future                 = 0,
    Option                 = 1,
    LastTaifexProuductType = 2,
};

enum MarketImpactResetMode
{
    Off             = 0,
    NoImpact        = 1,
    SingleTickReset = 2,
    LastResetMode   = 3,
};

enum TimerMode
{
    PacketEnd     = 0,
    AccurateTimer = 1,
    LastTimerMode = 2,
};

enum SimulationMode
{
    ImmediatelyFilled        = 0,
    FilledByFillModel        = 1,
    FilledByBookDiff         = 2,
    FilledByBookDiffAndTrade = 4,
    LastSimulationMode       = 5,
};

enum class OrderReportType
{
    OrderAccepted     = 0,
    OrderRejected     = 1,
    OrderCancelled    = 2,
    OrderCancelFailed = 3,
    OrderExecuted     = 4,
    OrderModified     = 5,
    OrderModifyFailed = 6,
};

struct RodOrderInfo
{
    bool is_aggressive_;
    bool side_;
    int  touch_price_;
    int  place_price_;
    int  qty_behind_place_price_;
    int  qty_at_touch_price_;
    int  qty_at_place_price_;
};

struct TaifexSimulationOrderReportMessage
{
    int             price_;
    int             qty_;
    int             leaves_qty_;
    int             order_no_;
    OrderReportSide side_;
    OrderReportType type_;
    Timestamp       processed_event_loop_time_;
};

enum OrderPosition
{
    Before      = 0,
    After       = 1,
    BeforeAfter = 2,
};

struct QueuePosition
{
    QueuePosition(int before, int after) : position_{before, after}, last_position_{0, 0}
    {
    }
    int position_[OrderPosition::BeforeAfter];
    int last_position_[OrderPosition::BeforeAfter];
};

struct TaifexOnExchOrder
{
    TaifexOnExchOrder() = default;
    TaifexOnExchOrder(int price, int original_qty, int remaining_qty, int order_no,
                      const TAIFEX_ORDER_SIDE &side, const Timestamp &order_sent_ts,
                      const Timestamp &order_received_by_exch_ts, const Timestamp &last_executed_ts,
                      const TAIFEX_ORDER_TIMEINFORCE &   timeinforce,
                      const TAIFEX_ORDER_POSITIONEFFECT &position_effect)
        : price_{price}
        , original_qty_{original_qty}
        , remaining_qty_{remaining_qty}
        , order_no_{order_no}
        , side_{side}
        , order_sent_ts_{order_sent_ts}
        , order_received_by_exch_ts_{order_received_by_exch_ts}
        , last_executed_ts_{last_executed_ts}
        , timeinforce_{timeinforce}
        , position_effect_{position_effect}
        , surpass_trade_numbers_{0}
        , is_need_to_be_cancelled_{false}
        , is_need_to_be_deleted_{false}
        , order_cancel_sent_ts_{Timestamp::invalid()}
    {
    }
    int price_;
    int original_qty_;
    int remaining_qty_;
    int order_no_;

    TAIFEX_ORDER_SIDE           side_;
    Timestamp                   order_sent_ts_;
    Timestamp                   order_received_by_exch_ts_;
    Timestamp                   last_executed_ts_;
    TAIFEX_ORDER_TIMEINFORCE    timeinforce_;
    TAIFEX_ORDER_POSITIONEFFECT position_effect_;

    int       surpass_trade_numbers_;
    bool      is_need_to_be_cancelled_;
    bool      is_need_to_be_deleted_;
    Timestamp order_cancel_sent_ts_;
};

struct NewOrderData
{
    NewOrderData(const char *pid, const int price, const int qty, const TAIFEX_ORDER_SIDE side,
                 const TAIFEX_ORDER_TIMEINFORCE    timeinforce,
                 const TAIFEX_ORDER_POSITIONEFFECT position_effect, const ProductType product_type)
        : pid_{pid}
        , price_{price}
        , qty_{qty}
        , side_{side}
        , timeinforce_{timeinforce}
        , position_effect_{position_effect}
        , product_type_{product_type}
    {
    }
    const char *                pid_;
    int                         price_;
    int                         qty_;
    TAIFEX_ORDER_SIDE           side_;
    TAIFEX_ORDER_TIMEINFORCE    timeinforce_;
    TAIFEX_ORDER_POSITIONEFFECT position_effect_;
    ProductType                 product_type_;
};

struct CancelOrderData
{
    CancelOrderData() = default;
    CancelOrderData(const int order_no, const char *pid, const TAIFEX_ORDER_SIDE side,
                    const ProductType type)
        : order_no_{order_no}
        , pid_{pid}
        , side_{side}
        , product_type_{type}
        , is_need_cancel_fail_{true}
        , cancel_sent_ts_{Timestamp::invalid()}
    {
    }

    int               order_no_;
    const char *      pid_;
    TAIFEX_ORDER_SIDE side_;
    ProductType       product_type_;
    bool              is_need_cancel_fail_;
    Timestamp         cancel_sent_ts_;
};

class TaifexSimulator : public BookDataListener
{
  public:
    TaifexSimulator(const GlobalConfiguration *config, const Symbol *symbol,
                    const MultiBookManager *multi_book_manager, Engine *engine,
                    TaifexSimulationOrderManager *order_manager);
    ~TaifexSimulator();

    void                    Init();
    std::array<int, AskBid> GetTouchPair(BookSide side, BookPrice price, BookPrice implied_bid,
                                         BookPrice implied_ask);

    void ProcessNewOrder(int orderno, int price, int qty, const TAIFEX_ORDER_SIDE &side,
                         const TAIFEX_ORDER_TIMEINFORCE &   timeInForce,
                         const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                         const Timestamp &                  event_loop_time);
    void NewOrder(const int orderno, const char *pid, int price, int qty, TAIFEX_ORDER_SIDE side,
                  TAIFEX_ORDER_TIMEINFORCE timeInForce, TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                  const Timestamp &received_event_loop_time);
    void CancelOrder(const int orderno, const char *pid, TAIFEX_ORDER_SIDE side,
                     bool is_need_cancel_fail, const Timestamp &received_event_loop_time);
    void NewDoubleOrder(const int orderno, const char *pid, int bidprice, int bidqty, int askprice,
                        int askqty, TAIFEX_ORDER_TIMEINFORCE timeInForce,
                        TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                        const Timestamp &           received_event_loop_time);
    void CancelDoubleOrder(const char *orderno, const char *pid,
                           const Timestamp &received_event_loop_time);

    void ModifyOrder(const char *orderno, const char *pid, int bidprice, int askprice,
                     const Timestamp &received_event_loop_time);

    void InsertOrderQueuePosition(TaifexOnExchOrder *order);
    void InsertOrderQueuePositionCallBack(const Timestamp event_loop_time,
                                          const Timestamp call_back_time, void *structure);
    void CleanOrderQueuePosition(const bool &side, const TaifexOnExchOrder *o);

    std::pair<int, int> GetOrderQueuePosition(const bool &side, const TaifexOnExchOrder *o);

    int  GetQtyTakenOnBook(const BookSide &side, int exec_price);
    void UpdateQtyTakenOnBook(const BookSide &side, int exec_price, int exec_qty);
    template <typename Side_T>
    void CleanQtyTakenOnBook()
    {
        const auto &side        = Side_T::GetSide();
        const auto &touch_price = touch_price_[side];
        for (auto it = qty_taken_from_book_total_[side].begin();
             it != qty_taken_from_book_total_[side].end();)
        {
            if (Side_T::IsCrossed(it->first, touch_price))
            {
                it = qty_taken_from_book_total_[side].erase(it);
            }
            else
            {
                break;
            }
        }
    }

    void InitBookCache(BookCache *book_cache);
    void RefreshBookCache(BookCache *book_cache, const bool &side);
    void CleanBookCache(BookCache *book_cache);

    inline void InsertOrderReport(int price, int qty, int leaves_qty, const TAIFEX_ORDER_SIDE &side,
                                  const TAIFEX_ORDER_TIMEINFORCE &   timeInForce,
                                  const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                                  const OrderReportType &type, int orderno,
                                  const Timestamp &event_loop_time)
    {
        auto o = GetOrderReportFromPool();
        FillInOrderReport(o, price, qty, leaves_qty, side, timeInForce, positionEffect, type,
                          orderno, event_loop_time);
        engine_->AddOneTimeTimer(event_loop_time,
                                 &alphaone::TaifexSimulator::InsertOrderReportCallBack, this, o);
    }
    inline void InsertOrderReportWithTimer(int price, int qty, int leaves_qty,
                                           const TAIFEX_ORDER_SIDE &          side,
                                           const TAIFEX_ORDER_TIMEINFORCE &   timeInForce,
                                           const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                                           const OrderReportType &type, int orderno,
                                           const Timestamp &event_loop_time)
    {
        auto o = GetOrderReportFromPool();
        FillInOrderReport(o, price, qty, leaves_qty, side, timeInForce, positionEffect, type,
                          orderno, event_loop_time);
        engine_->AddOneTimeTimer(event_loop_time,
                                 &alphaone::TaifexSimulator::InsertOrderReportCallBack, this, o);
    }

    void InsertOrderReportCallBack(const Timestamp event_loop_time, const Timestamp call_back_time,
                                   void *structure);

    inline void FillInOrderReport(TaifexSimulationOrderReportMessage *o, int price, int qty,
                                  int leaves_qty, const TAIFEX_ORDER_SIDE &side,
                                  const TAIFEX_ORDER_TIMEINFORCE &   timeInForce,
                                  const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                                  const OrderReportType &type, int orderno,
                                  const Timestamp &event_loop_time)
    {
        o->price_ = price;
        o->qty_   = qty;
        o->side_  = side == TAIFEX_ORDER_SIDE::BUY ? OrderReportSide::Buy : OrderReportSide::Sell;
        o->leaves_qty_                = leaves_qty;
        o->type_                      = type;
        o->order_no_                  = orderno;
        o->processed_event_loop_time_ = event_loop_time;
    }

    template <typename Side_T, bool is_from_passive>
    std::vector<std::pair<int, int>> AggressiveMatch(int price, int qty,
                                                     const TAIFEX_ORDER_TIMEINFORCE &tif)
    {
        std::vector<std::pair<int, int>> exec_pairs;

        const auto &check_side    = Side_T::other::GetSide();
        auto        current_level = book_->GetTouchLevel(check_side);
        if (BRANCH_UNLIKELY(!current_level))
            return exec_pairs;

        const auto &touch_price{static_cast<int>(std::nearbyint(current_level->price_ * pm_))};
        if (!Side_T::other::IsCrossed(price, touch_price))
            return exec_pairs;

        while (qty > 0)
        {
            const auto &c_price{static_cast<int>(std::nearbyint(current_level->price_ * pm_))};
            if (!Side_T::other::IsInnerOrEqual(c_price, price))
                break;

            auto c_qty{static_cast<int>(current_level->qty_)};
            if (is_considering_implied_ &&
                book_->GetImpliedOrderPrice(check_side) == current_level->price_)
                c_qty += book_->GetImpliedOrderQty(check_side);
            const auto &taken_qty{GetQtyTakenOnBook(check_side, c_price)};
            const auto  exec_qty{std::min(std::max(c_qty - taken_qty, 0), qty)};

            if (exec_qty)
            {
                qty -= exec_qty;
                if constexpr (is_from_passive)
                {
                    exec_pairs.emplace_back(price, exec_qty);
                    UpdateQtyTakenOnBook(check_side, price, exec_qty);
                }
                else
                {
                    exec_pairs.emplace_back(c_price, exec_qty);
                    UpdateQtyTakenOnBook(check_side, c_price, exec_qty);
                }
            }
            current_level = current_level->next_;
            if (!current_level)
                break;
        }

        if (tif == TAIFEX_ORDER_TIMEINFORCE::FOK && qty != 0)
        {
            for (const auto &[deduct_price, deduct_qty] : exec_pairs)
                UpdateQtyTakenOnBook(check_side, deduct_price, -deduct_qty);
            exec_pairs.clear();
        }

        return exec_pairs;
    }

    template <typename Side_T>
    std::vector<std::pair<int, int>> PassiveMatch(TaifexOnExchOrder *order, BookCache *book_cache)
    {
        std::vector<std::pair<int, int>> exec_pairs;

        const bool &check_side  = Side_T::GetSide();
        const auto &order_price = order->price_;
        auto        order_qty   = order->remaining_qty_;
        const auto &touch_level = book_->GetTouchLevel(check_side);
        if (BRANCH_UNLIKELY(!touch_level))
            return exec_pairs;

        const auto &touch_price = static_cast<int>(std::nearbyint(touch_level->price_ * pm_));
        if (!Side_T::IsInnerOrEqual(order_price, touch_price))
            return exec_pairs;

        auto touch_qty = static_cast<int>(touch_level->qty_);
        if (is_considering_implied_ && book_->GetImpliedOrderPrice(check_side) == touch_price)
            touch_qty += book_->GetImpliedOrderQty(check_side);

        // book diff match logic here
        // determine how much to exeucte, if not then the order remains in queue
        for (int i = 0; i < peek_num_of_levels_; ++i)
        {
            const auto &[prev_p, prev_q] = (*(*book_cache)[check_side])[i];
            if (Side_T::IsInner(touch_price, prev_p))
                break;

            if (Side_T::IsInnerOrEqual(order_price, prev_p) &&
                Side_T::IsInnerOrEqual(prev_p, touch_price))
            {
                // Get order queue position first, if not still many then break;
                const auto &[curr_queue_pos, last_queue_pos] =
                    GetOrderQueuePosition(check_side, order);
                if (curr_queue_pos >= 0)
                    break;
                const auto taken_qty = GetQtyTakenOnBook(check_side, prev_p);
                auto       exec_qty  = std::min(
                    order_qty, (prev_p == touch_price) ? std::max(prev_q - touch_qty - taken_qty, 0)
                                                              : std::max(prev_q - taken_qty, 0));
                exec_qty = last_queue_pos * curr_queue_pos < 0 ? std::min(-curr_queue_pos, exec_qty)
                                                               : exec_qty;
                exec_qty = mode_ == FilledByBookDiffAndTrade
                               ? std::min(exec_qty, GetTradesQtyFromPrice<Side_T>(
                                                        engine_->GetCurrentTime(), order_price))
                               : exec_qty;
                if (exec_qty != 0)
                {
                    exec_pairs.emplace_back(order_price, exec_qty);
                    UpdateQtyTakenOnBook(check_side, order_price, exec_qty);
                    order_qty -= exec_qty;
                }
            }
        }
        auto cross_exec_pairs =
            AggressiveMatch<Side_T, true>(order_price, order_qty, TAIFEX_ORDER_TIMEINFORCE::ROD);
        exec_pairs.insert(exec_pairs.end(), cross_exec_pairs.begin(), cross_exec_pairs.end());
        return exec_pairs;
    }

    template <typename Side_T>
    std::vector<std::pair<int, int>> GetExecPairsFromPrice(BookPrice price,
                                                           TaifexQty qty_to_be_executed)
    {
        std::vector<std::pair<int, int>> exec_pairs;

        const auto &side = Side_T::other::GetSide();
        auto        touch{book_->GetTouchLevel(side)};
        while (qty_to_be_executed)
        {
            if (BRANCH_UNLIKELY(!touch || !touch->price_))
                break;

            const auto &exec_price{static_cast<int>(std::nearbyint(touch->price_ * pm_))};
            auto        qty{touch->qty_};
            if (is_considering_implied_ && book_->GetImpliedOrderPrice(side) == touch->price_)
                qty += book_->GetImpliedOrderQty(side);
            const auto &taken_qty{GetQtyTakenOnBook(side, exec_price)};
            const auto  exec_qty{
                std::min(static_cast<int>(std::nearbyint(std::max(0., qty - taken_qty))),
                         qty_to_be_executed)};

            if (!exec_qty)
            {
                touch = touch->next_;
                continue;
            }
            if (Side_T::other::IsInner(price, touch->price_))
                return exec_pairs;

            exec_pairs.emplace_back(exec_price, exec_qty);
            UpdateQtyTakenOnBook(side, exec_price, exec_qty);
            qty_to_be_executed -= exec_qty;
            touch = touch->next_;
        }
        return exec_pairs;
    }

    template <bool IsNeedCancel>
    std::pair<bool, BookQty> CleanPassiveOrderResources(int              order_no,
                                                        const Timestamp &received_event_loop_time,
                                                        BookSide         side)
    {
        BookQty qty{0.};
        if (auto it = order_no_to_on_exch_orders_map_.find(order_no);
            it != order_no_to_on_exch_orders_map_.end())
        {
            // set up cancel flag
            qty = it->second->remaining_qty_;
            if (it->second->remaining_qty_ > 0)
            {
                it->second->is_need_to_be_deleted_ = true;
                if constexpr (IsNeedCancel)
                {
                    it->second->is_need_to_be_cancelled_ = true;
                    it->second->order_cancel_sent_ts_    = received_event_loop_time;
                }
                else
                {
                }
            }

            // clean queue position
            CleanOrderQueuePosition(side, it->second);

            // clean By Book
            if (auto bcit = order_to_prev_book_cache_map_.find(it->second);
                bcit != order_to_prev_book_cache_map_.end())
            {
                CleanBookCache(bcit->second);
                order_to_prev_book_cache_map_.erase(bcit);
            }

            order_no_to_on_exch_orders_map_.erase(it);
            return {true, qty};
        }
        return {false, qty};
    }

    void CleanPassiveOrderResourcesCallBack(const Timestamp event_loop_time,
                                            const Timestamp call_back_time, void *structure);

    void InsertIocReport(const Timestamp &event_loop_time, int orderno, int price, int qty,
                         const TAIFEX_ORDER_SIDE &side, const TAIFEX_ORDER_TIMEINFORCE &timeInForce,
                         const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                         std::vector<std::pair<int, int>> * exec_pairs);
    void InsertFokReport(const Timestamp &event_loop_time, int orderno, int price, int qty,
                         const TAIFEX_ORDER_SIDE &side, const TAIFEX_ORDER_TIMEINFORCE &timeInForce,
                         const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                         std::vector<std::pair<int, int>> * exec_pairs);
    int  InsertRodReport(const Timestamp &event_loop_time, int orderno, int price, int qty,
                         const TAIFEX_ORDER_SIDE &side, const TAIFEX_ORDER_TIMEINFORCE &timeInForce,
                         const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                         std::vector<std::pair<int, int>> * exec_pairs);

    void ProcessAggressiveOrderQueue(const Timestamp &event_loop_time);
    void ProcessPassiveOrderQueue(const Timestamp &event_loop_time);
    void ProcessOrderQueueCallBack(const Timestamp event_loop_time, const Timestamp call_back_time,
                                   void *structure);

    // pre-book tick events
    void OnPreBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
    {
    }

    // post-book tick events
    void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o);
    void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o);
    void OnPostBookModifyWithPrice(const Timestamp                       event_loop_time,
                                   const BookDataMessageModifyWithPrice *o)
    {
    }
    void OnPostBookModifyWithQty(const Timestamp                     event_loop_time,
                                 const BookDataMessageModifyWithQty *o)
    {
    }
    void OnPostBookSnapshot(const Timestamp event_loop_time, const BookDataMessageSnapshot *o)
    {
    }
    // trade tick events
    void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o);

    // tick event after all event with the same timestamp has already been updated
    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o);

    // for production use
    void OnSparseStop(const Timestamp &event_loop_time, const BookDataMessageSparseStop *o)
    {
    }

    void OnTPrice(const Timestamp &event_loop_time, const MarketDataMessage *t)
    {
    }

    template <typename Side_T>
    int GetTradesQtyFromPrice(const Timestamp &event_loop_time, int price)
    {
        const auto &side = Side_T::GetSide();
        // clean old trades
        while (!possible_filled_trades_[side].empty() &&
               event_loop_time - possible_filled_trades_[side].front().second >
                   possible_trades_window_)
        {
            possible_filled_trades_[side].pop_front();
        }

        int qty{0};
        for (const auto &c : possible_filled_trades_[side])
        {
            if (Side_T::IsInnerOrEqual(price, c.first.first))
            {
                qty += c.first.second;
            }
        }
        return qty;
    }

    std::vector<TaifexSimulationOrderReportMessage *> *GetOrderReports()
    {
        return &order_reports_;
    }

    std::string SimulationModeToString(const SimulationMode &mode)
    {
        switch (mode)
        {
        case SimulationMode::ImmediatelyFilled:
            return "ImmediatelyFilled";
        case SimulationMode::FilledByFillModel:
            return "FilledByFillModel";
        case SimulationMode::FilledByBookDiff:
            return "FilledByBookDiff";
        case SimulationMode::FilledByBookDiffAndTrade:
            return "FilledByBookDiffAndTrade";
        case SimulationMode::LastSimulationMode:
            return "LastSimulationMode";
        default:
            return "Not Defined";
        }
    }

    std::string TimerModeToString(const TimerMode &mode)
    {
        switch (mode)
        {
        case TimerMode::PacketEnd:
            return "PacketEnd";
        case TimerMode::AccurateTimer:
            return "AccurateTimer";
        default:
            return "Not Defined";
        }
    }

    std::string MarketImpactResetModeToString(const MarketImpactResetMode &mode)
    {
        switch (mode)
        {
        case MarketImpactResetMode::Off:
            return "Off";
        case MarketImpactResetMode::NoImpact:
            return "NoImapct";
        case MarketImpactResetMode::SingleTickReset:
            return "SingleTickReset";
        default:
            return "Not Defined";
        }
    }

    const Symbol *GetSymbol()
    {
        return symbol_;
    }

    boost::pool<boost::default_user_allocator_malloc_free> *GetOrderReportsPool()
    {
        return order_reports_pool_;
    }
    boost::pool<boost::default_user_allocator_malloc_free> *GetOutstandingOrdersPool()
    {
        return on_exch_orders_pool_;
    }
    TaifexSimulationOrderReportMessage *GetOrderReportFromPool()
    {
        return GetObjFromPool<TaifexSimulationOrderReportMessage>(order_reports_pool_);
    }
    TaifexOnExchOrder *GetOnExchOrderFromPool()
    {
        return GetObjFromPool<TaifexOnExchOrder>(on_exch_orders_pool_);
    }
    TaifexOnExchOrder *GetOnExchOrderFromPool(int price, int original_qty, int remaining_qty,
                                              int order_no, const TAIFEX_ORDER_SIDE &side,
                                              const Timestamp &                  order_sent_ts,
                                              const Timestamp &                  exch_received_ts,
                                              const Timestamp &                  last_executed_ts,
                                              const TAIFEX_ORDER_TIMEINFORCE &   timeinforce,
                                              const TAIFEX_ORDER_POSITIONEFFECT &position_effect)
    {
        return GetObjFromPool<TaifexOnExchOrder, int, int, int, int, TAIFEX_ORDER_SIDE, Timestamp,
                              Timestamp, Timestamp, TAIFEX_ORDER_TIMEINFORCE,
                              TAIFEX_ORDER_POSITIONEFFECT>(
            on_exch_orders_pool_, price, original_qty, remaining_qty, order_no, side, order_sent_ts,
            exch_received_ts, last_executed_ts, timeinforce, position_effect);
    }
    BookCache *GetBookCacheFromPool()
    {
        return GetObjFromPool<BookCache>(prev_book_pool_);
    }

    CancelOrderData *GetCancelOrderDataFromPool()
    {
        return GetObjFromPool<CancelOrderData>(cancel_info_pool_);
    }

    void SetBook(MarketByOrderBook *book)
    {
        const_cast<MarketByOrderBook *&>(book_) = book;
    }

  private:
    const GlobalConfiguration *   config_;
    nlohmann::json                json_;
    const Symbol *                symbol_;
    SimulationMode                mode_;
    const MultiBookManager *      multi_book_manager_;
    const int                     pm_;
    TimerMode                     timer_mode_;
    MarketImpactResetMode         market_impact_reset_mode_;
    Duration                      forward_delay_[PassiveAggressive];
    Duration                      backward_delay_[PassiveAggressive];
    const Duration                rod_report_delay_;
    const Duration                cancel_delay_;
    const Duration                cancel_report_delay_;
    const Duration                double_order_delay_;
    const int                     peek_num_of_levels_;
    const Duration                match_time_;
    bool                          is_optimistic_when_filled_by_bookdiff_;
    bool                          is_optimistic_of_queue_position_;
    const MarketByOrderBook *     book_;
    int                           numbers_of_trade_to_check_;
    const Duration                possible_trades_window_;
    Engine *                      engine_;
    TaifexSimulationOrderManager *order_manager_;
    bool                          is_considering_implied_;

    const double double_order_percentage_;
    const int    double_order_qty_;
    // config parameters
    std::unordered_map<std::string, SimulationMode>        simulation_mode_string_to_enum_map_;
    std::unordered_map<std::string, TimerMode>             timer_mode_string_to_enum_map_;
    std::unordered_map<std::string, MarketImpactResetMode> market_reset_to_enum_map_;

    // two kind of order queue to process
    std::queue<std::pair<TaifexOnExchOrder *, std::array<int, AskBid>>>
                                   aggressive_order_size_pairs_;
    std::list<TaifexOnExchOrder *> passive_orders_queue_;

    // mem pool
    boost::pool<boost::default_user_allocator_malloc_free> *order_reports_pool_;
    boost::pool<boost::default_user_allocator_malloc_free> *on_exch_orders_pool_;
    boost::pool<boost::default_user_allocator_malloc_free> *prev_book_pool_;
    boost::pool<boost::default_user_allocator_malloc_free> *cancel_info_pool_;

    // communicate with order manager
    std::vector<TaifexSimulationOrderReportMessage *> order_reports_;

    // map to deal with cancel and delay book
    std::unordered_map<TaifexOnExchOrder *, BookCache *> order_to_prev_book_cache_map_;
    std::unordered_map<int, TaifexOnExchOrder *>         order_no_to_on_exch_orders_map_;

    // orderbook interaction statistic
    std::map<int, int, dynamic_compare> qty_taken_from_book_total_[AskBid];
    std::map<int, std::map<const TaifexOnExchOrder *, QueuePosition>, dynamic_compare>
        order_queue_position_[AskBid];

    std::deque<std::pair<std::pair<int, int>, Timestamp>> possible_filled_trades_[AskBid];
    std::map<int, BookQty>                                trade_pairs_[AskBid];

    const int      qty_in_front_[AskBid];
    const Duration order_shift_;
    int            touch_price_[AskBid];
    int            double_order_no_;

    // delay related
    DelayReader *     delay_reader_{nullptr};
    long double       calc_time_;
    int64_t           curr_seqno_;
    const std::string delay_root_path_;
    const std::string delay_file_name_;

    const bool is_force_passive_;
};

class TaifexSimulationOrderManager : public TaifexOrderManagerBase
{
  public:
    TaifexSimulationOrderManager(MultiBookManager *multi_book_manager,
                                 ObjectManager *object_manager, Engine *engine,
                                 size_t config_index = 0);
    ~TaifexSimulationOrderManager();

    void SetUpSimulatorListeners();

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

    TaifexOrderStatus *NewOptionOrder(const char *pid, int price, int qty, TAIFEX_ORDER_SIDE side,
                                      TAIFEX_ORDER_TIMEINFORCE    timeInForce,
                                      TAIFEX_ORDER_POSITIONEFFECT positionEffect, int sessionIndex,
                                      int subSessionIndex, unsigned int account, char accountFlag,
                                      const char *pUserDefine = nullptr);

    void CancelOptionOrder(const char *orderno, const char *pid, TAIFEX_ORDER_SIDE side,
                           int sessionIndex, int subSessionIndex,
                           const char *pUserDefine = nullptr);

    TaifexOrderStatus *NewOptionDoubleOrder(const char *pid, int bidprice, int bidqty, int askprice,
                                            int askqty, TAIFEX_ORDER_TIMEINFORCE timeInForce,
                                            TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                                            int sessionIndex, int subSessionIndex,
                                            unsigned int account, char accountFlag,
                                            const char *pUserDefine = nullptr);

    void ModifyOptionDoubleOrder(const char *orderno, const char *pid, int bidprice, int askprice,
                                 int sessionIndex, int subSessionIndex,
                                 const char *pUserDefine = nullptr);

    void CancelOptionDoubleOrder(const char *orderno, const char *pid, int sessionIndex,
                                 int subSessionIndex, const char *pUserDefine = nullptr);

    void NewOrderCallBack(const Timestamp event_loop_time, const Timestamp call_back_time,
                          void *structure);
    void CancelOrderCallBack(const Timestamp event_loop_time, const Timestamp call_back_time,
                             void *structure);

    TaifexOrderStatus *GetOrderStatusFromPoolWithId(int id)
    {
        return GetObjFromPool<TaifexOrderStatus>(order_status_pools_[id]);
    }

    void CleanOrderStatus(int order_no, int id);

    inline int GetOrderCount()
    {
        return order_count_;
    }

    void Process(const Timestamp &event_loop_time);
    void ProcessCallBack(const Timestamp event_loop_time, const Timestamp call_back_time,
                         void *structure);

  private:
    MultiBookManager *                        multi_book_manager_;
    ObjectManager *                           object_manager_;
    const std::vector<const Symbol *> &       universe_;
    const size_t                              symbol_numbers_;
    std::vector<TaifexOrderReportListener *> &order_report_listeners_;
    std::vector<TaifexSimulator *>            taifex_simulators_;
    std::unordered_map<std::string, int>      symbol_to_id_maps_;
    int                                       order_count_;
    Engine *                                  engine_;

    std::vector<boost::pool<boost::default_user_allocator_malloc_free> *> order_status_pools_;

    char curr_orderno_[TMP_ORDNO_LEN + 1];
};


}  // namespace alphaone
#endif
