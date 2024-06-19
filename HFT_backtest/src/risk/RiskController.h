#ifndef _RISKCONTROLLER_H_
#define _RISKCONTROLLER_H_

#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/platform/manager/MultiBookManager.h"
#include "risk/RiskStatus.h"

namespace alphaone
{
class RiskController
{
  public:
    RiskController(const MultiBookManager *multi_book_manager, const std::string &name);
    RiskController(const RiskController &) = delete;
    RiskController &operator=(const RiskController &) = delete;
    RiskController(RiskController &&)                 = delete;
    RiskController &operator=(RiskController &&) = delete;

    ~RiskController() = default;

    bool                  IsSoftStopped(const Timestamp &event_loop_time) const;
    bool                  IsHardStopped(const Timestamp &event_loop_time) const;
    RiskStatus            GetRiskStatus() const;
    size_t                GetTotalOrders(const Timestamp &event_loop_time) const;
    size_t                GetOrdersThisSec(const Timestamp &event_loop_time);
    size_t                GetOrdersThisMin(const Timestamp &event_loop_time);
    size_t                GetBouncesThisMin(const Timestamp &event_loop_time);
    size_t                GetRepeatedOrdersThisMin(const Timestamp &event_loop_time);
    const nlohmann::json &GetRiskParameters() const;

    void ResetCount(const Timestamp &event_loop_time);

    RiskStatus CheckOrdersPerSecond(const Timestamp &event_loop_time, const Symbol *symbol,
                                    const BookSide &side, const BookPrice &price,
                                    const BookQty &qty);
    RiskStatus CheckOrdersPerMinute(const Timestamp &event_loop_time, const Symbol *symbol,
                                    const BookSide &side, const BookPrice &price,
                                    const BookQty &qty);
    RiskStatus CheckOrderPrice(const Timestamp &event_loop_time, const Symbol *symbol,
                               const BookSide &side, const BookPrice &price, const BookQty &qty);
    RiskStatus CheckTotalOrders(const Timestamp &event_loop_time, const Symbol *symbol,
                                const BookSide &side, const BookPrice &price, const BookQty &qty);
    RiskStatus CheckTooFarFromLastTrade(const Timestamp &event_loop_time, const Symbol *symbol,
                                        const BookSide &side, const BookPrice &price,
                                        const BookQty &qty);
    RiskStatus CheckTooFarCrossTouch(const Timestamp &event_loop_time, const Symbol *symbol,
                                     const BookSide &side, const BookPrice &price,
                                     const BookQty &qty);
    RiskStatus CheckBouncesPerMinute(const Timestamp &event_loop_time, const Symbol *symbol,
                                     const BookSide &side, const BookPrice &price,
                                     const BookQty &qty);
    RiskStatus CheckRepeatedOrdersPerMinute(const Timestamp &event_loop_time, const Symbol *symbol,
                                            const BookSide &side, const BookPrice &price,
                                            const BookQty &qty);

    template <RiskControllerType type>
    RiskStatus CheckRiskStatus(const Timestamp &event_loop_time, const Symbol *symbol,
                               const BookSide &side, const BookPrice &price, const BookQty &qty);

  private:
    constexpr static Duration ONE_SEC{Duration::from_sec(static_cast<int64_t>(1))};
    constexpr static Duration ONE_MIN{Duration::from_min(static_cast<int64_t>(1))};

    const std::string       name_;
    const MultiBookManager *multi_book_manager_;

    const size_t SOFT_MAX_ORDERS_PER_SEC_;
    const size_t SOFT_MAX_ORDERS_PER_MIN_;
    const double SOFT_MAX_DISTANCE_FROM_LAST_TRADE_;
    const double SOFT_MAX_DISTANCE_CROSS_TOUCH_;
    const size_t SOFT_MAX_BOUNCES_PER_MIN_;
    const size_t SOFT_MAX_REPEATED_ORDERS_PER_MIN_;

    const size_t HARD_MAX_TOTAL_ORDERS_;

    const size_t HARD_MAX_ORDERS_PER_SEC_;
    const size_t HARD_MAX_ORDERS_PER_MIN_;
    const double HARD_MAX_DISTANCE_FROM_LAST_TRADE_;
    const double HARD_MAX_DISTANCE_CROSS_TOUCH_;
    const size_t HARD_MAX_BOUNCES_PER_MIN_;
    const size_t HARD_MAX_REPEATED_ORDERS_PER_MIN_;

    Timestamp last_check_sec_;
    Timestamp last_check_min_;
    Timestamp soft_triggered_;

    mutable RiskStatus risk_status_;

    size_t total_orders_;
    size_t orders_this_sec_;
    size_t orders_this_min_;
    size_t bounces_this_min_;
    size_t repeated_orders_this_min_;

    BookSide  last_0_sent_order_side_;
    BookPrice last_0_sent_order_price_;
    BookQty   last_0_sent_order_qty_;
    BookSide  last_1_sent_order_side_;
    BookPrice last_1_sent_order_price_;
    BookQty   last_1_sent_order_qty_;

    nlohmann::json risk_parameters_;
};

inline bool RiskController::IsSoftStopped(const Timestamp &event_loop_time) const
{
    if (risk_status_ > RiskStatus::Good && risk_status_ < RiskStatus::HardStop_TotalOrders)
    {
        if (event_loop_time < soft_triggered_ + ONE_MIN)
        {
            return true;
        }
        else
        {
            risk_status_ = RiskStatus::Good;
            return false;
        }
    }
    else
    {
        return false;
    }
}

inline bool RiskController::IsHardStopped(const Timestamp &event_loop_time) const
{
    return risk_status_ >= RiskStatus::HardStop_TotalOrders;
}

inline RiskStatus RiskController::GetRiskStatus() const
{
    return risk_status_;
}

inline size_t RiskController::GetTotalOrders(const Timestamp &event_loop_time) const
{
    return total_orders_;
}

inline size_t RiskController::GetOrdersThisSec(const Timestamp &event_loop_time)
{
    ResetCount(event_loop_time);
    return orders_this_sec_;
}

inline size_t RiskController::GetOrdersThisMin(const Timestamp &event_loop_time)
{
    ResetCount(event_loop_time);
    return orders_this_min_;
}

inline size_t RiskController::GetBouncesThisMin(const Timestamp &event_loop_time)
{
    ResetCount(event_loop_time);
    return bounces_this_min_;
}

inline size_t RiskController::GetRepeatedOrdersThisMin(const Timestamp &event_loop_time)
{
    ResetCount(event_loop_time);
    return repeated_orders_this_min_;
}

inline void RiskController::ResetCount(const Timestamp &event_loop_time)
{
    if (!last_check_sec_.is_valid() or event_loop_time >= last_check_sec_ + ONE_SEC)
    {
        last_check_sec_  = event_loop_time;
        orders_this_sec_ = 0;
    }

    if (!last_check_min_.is_valid() or event_loop_time >= last_check_min_ + ONE_MIN)
    {
        last_check_min_           = event_loop_time;
        orders_this_min_          = 0;
        bounces_this_min_         = 0;
        repeated_orders_this_min_ = 0;
    }
}

// Check OrdersPerSecond
inline RiskStatus RiskController::CheckOrdersPerSecond(const Timestamp &event_loop_time,
                                                       const Symbol *symbol, const BookSide &side,
                                                       const BookPrice &price, const BookQty &qty)
{
    ++orders_this_sec_;
    if (orders_this_sec_ >= HARD_MAX_ORDERS_PER_SEC_)
    {
        SPDLOG_WARN("{} HardStop initiated due to {} count={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::HardStop_OrdersPerSecond), orders_this_sec_);
        return RiskStatus::HardStop_OrdersPerSecond;
    }
    else if (orders_this_sec_ >= SOFT_MAX_ORDERS_PER_SEC_)
    {
        std::stringstream ss;
        SPDLOG_WARN("{} SoftStop initiated due to {} count={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::SoftStop_OrdersPerSecond), orders_this_sec_);
        soft_triggered_ = event_loop_time;

        return RiskStatus::SoftStop_OrdersPerSecond;
    }
    return RiskStatus::Good;
}

// Check OrdersPerMinute
inline RiskStatus RiskController::CheckOrdersPerMinute(const Timestamp &event_loop_time,
                                                       const Symbol *symbol, const BookSide &side,
                                                       const BookPrice &price, const BookQty &qty)
{
    ++orders_this_min_;
    if (orders_this_min_ >= HARD_MAX_ORDERS_PER_MIN_)
    {
        SPDLOG_WARN("{} HardStop initiated due to {} count={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::HardStop_OrdersPerMinute), orders_this_min_);
        return RiskStatus::HardStop_OrdersPerMinute;
    }
    else if (orders_this_min_ >= SOFT_MAX_ORDERS_PER_MIN_)
    {
        SPDLOG_WARN("{} SoftStop initiated due to {} count={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::SoftStop_OrdersPerMinute), orders_this_min_);
        soft_triggered_ = event_loop_time;

        return RiskStatus::SoftStop_OrdersPerMinute;
    }
    return RiskStatus::Good;
}

// Check OrderPrice
inline RiskStatus RiskController::CheckOrderPrice(const Timestamp &event_loop_time,
                                                  const Symbol *symbol, const BookSide &side,
                                                  const BookPrice &price, const BookQty &qty)
{
    if (price <= 0.0)
    {
        SPDLOG_WARN("{} HardStop initiated due to {} price={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::HardStop_OrderPrice), price);
        return RiskStatus::HardStop_OrderPrice;
    }
    return RiskStatus::Good;
}

// Check TotalOrders
inline RiskStatus RiskController::CheckTotalOrders(const Timestamp &event_loop_time,
                                                   const Symbol *symbol, const BookSide &side,
                                                   const BookPrice &price, const BookQty &qty)
{
    ++total_orders_;
    if (total_orders_ >= HARD_MAX_TOTAL_ORDERS_)
    {
        SPDLOG_WARN("{} HardStop initiated due to {} count={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::HardStop_TotalOrders), total_orders_);
        return RiskStatus::HardStop_TotalOrders;
    }
    return RiskStatus::Good;
}

// Check TooFarFromLastTrade
inline RiskStatus RiskController::CheckTooFarFromLastTrade(const Timestamp &event_loop_time,
                                                           const Symbol *   symbol,
                                                           const BookSide & side,
                                                           const BookPrice &price,
                                                           const BookQty &  qty)
{
    const BookPrice last_trade_price{multi_book_manager_->GetBook(symbol)->GetLastTradePrice() *
                                     symbol->GetDecimalConverter()};
    if (side == BID)
    {
        if (price > (1.0 + HARD_MAX_DISTANCE_FROM_LAST_TRADE_) * last_trade_price)
        {
            SPDLOG_WARN("{} HardStop initiated due to {} side=BID price={} last_trade_price={}",
                        event_loop_time,
                        FromRiskStatusToString(RiskStatus::HardStop_TooFarFromLastTrade), price,
                        last_trade_price);
            return RiskStatus::HardStop_TooFarFromLastTrade;
        }
        else if (price > (1.0 + SOFT_MAX_DISTANCE_FROM_LAST_TRADE_) * last_trade_price)
        {
            SPDLOG_WARN("{} SoftStop initiated due to {} side=BID price={} last_trade_price={}",
                        event_loop_time,
                        FromRiskStatusToString(RiskStatus::SoftStop_TooFarFromLastTrade), price,
                        last_trade_price);
            soft_triggered_ = event_loop_time;

            return RiskStatus::SoftStop_TooFarFromLastTrade;
        }
    }
    else
    {
        if (price < (1.0 - HARD_MAX_DISTANCE_FROM_LAST_TRADE_) * last_trade_price)
        {
            SPDLOG_WARN("{} HardStop initiated due to {} side=ASK price={} last_trade_price={}",
                        event_loop_time,
                        FromRiskStatusToString(RiskStatus::HardStop_TooFarFromLastTrade), price,
                        last_trade_price);
            return RiskStatus::HardStop_TooFarFromLastTrade;
        }
        else if (price < (1.0 - SOFT_MAX_DISTANCE_FROM_LAST_TRADE_) * last_trade_price)
        {
            SPDLOG_WARN("{} SoftStop initiated due to {} side=ASK price={} last_trade_price={}",
                        event_loop_time,
                        FromRiskStatusToString(RiskStatus::SoftStop_TooFarFromLastTrade), price,
                        last_trade_price);
            soft_triggered_ = event_loop_time;

            return RiskStatus::SoftStop_TooFarFromLastTrade;
        }
    }

    return RiskStatus::Good;
}

// Check TooFarCrossTouch
inline RiskStatus RiskController::CheckTooFarCrossTouch(const Timestamp &event_loop_time,
                                                        const Symbol *symbol, const BookSide &side,
                                                        const BookPrice &price, const BookQty &qty)
{
    if (side == BID)
    {
        const BookPrice touch{multi_book_manager_->GetBook(symbol)->GetPrice(ASK) *
                              symbol->GetDecimalConverter()};
        if (price > (1.0 + HARD_MAX_DISTANCE_CROSS_TOUCH_) * touch)
        {
            SPDLOG_WARN(
                "{} HardStop initiated due to {} side=BID price={} touch={}", event_loop_time,
                FromRiskStatusToString(RiskStatus::HardStop_TooFarCrossTouch), price, touch);
            return RiskStatus::HardStop_TooFarCrossTouch;
        }
        else if (price > (1.0 + SOFT_MAX_DISTANCE_CROSS_TOUCH_) * touch)
        {
            SPDLOG_WARN(
                "{} SoftStop initiated due to {} side=BID price={} touch={}", event_loop_time,
                FromRiskStatusToString(RiskStatus::SoftStop_TooFarCrossTouch), price, touch);
            soft_triggered_ = event_loop_time;

            return RiskStatus::SoftStop_TooFarCrossTouch;
        }
    }
    else
    {
        const BookPrice touch{multi_book_manager_->GetBook(symbol)->GetPrice(BID) *
                              symbol->GetDecimalConverter()};
        if (price < (1.0 - HARD_MAX_DISTANCE_CROSS_TOUCH_) * touch)
        {
            SPDLOG_WARN(
                "{} HardStop initiated due to {} side=ASK price={} touch={}", event_loop_time,
                FromRiskStatusToString(RiskStatus::HardStop_TooFarCrossTouch), price, touch);
            return RiskStatus::HardStop_TooFarCrossTouch;
        }
        else if (price < (1.0 - SOFT_MAX_DISTANCE_CROSS_TOUCH_) * touch)
        {
            SPDLOG_WARN(
                "{} SoftStop initiated due to {} side=ASK price={} touch={}", event_loop_time,
                FromRiskStatusToString(RiskStatus::SoftStop_TooFarCrossTouch), price, touch);
            soft_triggered_ = event_loop_time;

            return RiskStatus::SoftStop_TooFarCrossTouch;
        }
    }

    return RiskStatus::Good;
}

// Check BouncesPerMinute
inline RiskStatus RiskController::CheckBouncesPerMinute(const Timestamp &event_loop_time,
                                                        const Symbol *symbol, const BookSide &side,
                                                        const BookPrice &price, const BookQty &qty)
{
    if (side == BID)
    {
        if (last_0_sent_order_side_ == ASK && last_0_sent_order_price_ < price)
        {
            ++bounces_this_min_;
        }
    }
    else
    {
        if (last_0_sent_order_side_ == BID && last_0_sent_order_price_ > price)
        {
            ++bounces_this_min_;
        }
    }

    if (bounces_this_min_ >= HARD_MAX_BOUNCES_PER_MIN_)
    {
        SPDLOG_WARN("{} HardStop initiated due to {} count={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::HardStop_BouncesPerMinute),
                    bounces_this_min_);
        return RiskStatus::HardStop_BouncesPerMinute;
    }
    else if (bounces_this_min_ >= SOFT_MAX_BOUNCES_PER_MIN_)
    {
        SPDLOG_WARN("{} SoftStop initiated due to {} count={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::SoftStop_BouncesPerMinute),
                    bounces_this_min_);
        soft_triggered_ = event_loop_time;

        return RiskStatus::SoftStop_BouncesPerMinute;
    }

    return RiskStatus::Good;
}

// Check RepeatedOrdersPerMinute
inline RiskStatus RiskController::CheckRepeatedOrdersPerMinute(const Timestamp &event_loop_time,
                                                               const Symbol *   symbol,
                                                               const BookSide & side,
                                                               const BookPrice &price,
                                                               const BookQty &  qty)
{
    if ((side == last_0_sent_order_side_ && price == last_0_sent_order_price_ &&
         qty == last_0_sent_order_qty_) ||
        (side == last_1_sent_order_side_ && price == last_1_sent_order_price_ &&
         qty == last_1_sent_order_qty_))
    {
        ++repeated_orders_this_min_;
    }

    if (repeated_orders_this_min_ >= HARD_MAX_REPEATED_ORDERS_PER_MIN_)
    {
        SPDLOG_WARN("{} HardStop initiated due to {} count={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::HardStop_RepeatedOrdersPerMinute),
                    repeated_orders_this_min_);
        return RiskStatus::HardStop_RepeatedOrdersPerMinute;
    }
    else if (repeated_orders_this_min_ >= SOFT_MAX_REPEATED_ORDERS_PER_MIN_)
    {
        SPDLOG_WARN("{} SoftStop initiated due to {} count={}", event_loop_time,
                    FromRiskStatusToString(RiskStatus::SoftStop_RepeatedOrdersPerMinute),
                    repeated_orders_this_min_);
        soft_triggered_ = event_loop_time;

        return RiskStatus::SoftStop_RepeatedOrdersPerMinute;
    }

    return RiskStatus::Good;
}

template <RiskControllerType type>
RiskStatus RiskController::CheckRiskStatus(const Timestamp &event_loop_time, const Symbol *symbol,
                                           const BookSide &side, const BookPrice &price,
                                           const BookQty &qty)
{
    // Check existing hardstop
    if (IsHardStopped(event_loop_time))
    {
        SPDLOG_WARN("{} Hard stop in place, ignoring further orders", event_loop_time);
        return risk_status_;
    }

    if (IsSoftStopped(event_loop_time))
    {
        if (event_loop_time < soft_triggered_ + ONE_MIN)
        {
            SPDLOG_WARN("{} Soft stop in place, ignoring further orders for one minute",
                        event_loop_time);
            return risk_status_;
        }
        else
        {
            risk_status_ = RiskStatus::Good;
        }
    }

    // Reset counts
    ResetCount(event_loop_time);

    if constexpr (type == RiskControllerType::Gaia)
    {
        // Check OrdersPerSecond
        if (const RiskStatus status{
                CheckOrdersPerSecond(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check OrdersPerMinute
        if (const RiskStatus status{
                CheckOrdersPerMinute(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check OrderPrice
        if (const RiskStatus status{CheckOrderPrice(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check TotalOrders
        if (const RiskStatus status{CheckTotalOrders(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check TooFarFromLastTrade
        if (const RiskStatus status{
                CheckTooFarFromLastTrade(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check TooFarCrossTouch
        if (const RiskStatus status{
                CheckTooFarCrossTouch(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check BouncesPerMinute
        if (const RiskStatus status{
                CheckBouncesPerMinute(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check RepeatedOrdersPerMinute
        if (const RiskStatus status{
                CheckRepeatedOrdersPerMinute(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }
    }
    else
    {
        // Check OrderPrice
        if (const RiskStatus status{CheckOrderPrice(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check TooFarFromLastTrade
        if (const RiskStatus status{
                CheckTooFarFromLastTrade(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check BouncesPerMinute
        if (const RiskStatus status{
                CheckBouncesPerMinute(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }

        // Check RepeatedOrdersPerMinute
        if (const RiskStatus status{
                CheckRepeatedOrdersPerMinute(event_loop_time, symbol, side, price, qty)};
            status != RiskStatus::Good)
        {
            return (risk_status_ = status);
        }
    }

    last_1_sent_order_side_ = last_0_sent_order_side_;
    last_0_sent_order_side_ = side;

    last_1_sent_order_price_ = last_0_sent_order_price_;
    last_0_sent_order_price_ = price;

    last_1_sent_order_qty_ = last_0_sent_order_qty_;
    last_0_sent_order_qty_ = qty;

    // Good
    return (risk_status_ = RiskStatus::Good);
}
}  // namespace alphaone

#endif
