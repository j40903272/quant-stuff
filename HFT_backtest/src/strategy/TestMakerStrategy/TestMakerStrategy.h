#ifndef _TESTMAKER_H_
#define _TESTMAKER_H_

#include "infrastructure/common/math/Math.h"
#include "infrastructure/common/side/Side.h"
#include "infrastructure/common/util/Order.h"
#include "infrastructure/platform/book/MarketByOrderBook.h"
#include "infrastructure/platform/engine/Engine.h"
#include "infrastructure/platform/manager/OutstandingOrderManager.h"
#include "infrastructure/platform/metric/MetricReporter.h"
#include "risk/RiskController.h"
#include "strategy/Strategy.h"
#include "strategy/StrategyFactory.h"

#include <unordered_set>

namespace alphaone
{
class TestMaker : public Tactic
{
  public:
    using Tactic::OnAccepted;
    using Tactic::OnCancelFailed;
    using Tactic::OnCancelled;
    using Tactic::OnDropOrder;
    using Tactic::OnExecuted;
    using Tactic::OnFastReport;
    using Tactic::OnRejectByServer;
    using Tactic::OnRejected;
    TestMaker(Ensemble *ensemble, const nlohmann::json node);
    TestMaker(const TestMaker &) = delete;
    TestMaker &operator=(const TestMaker &) = delete;

    ~TestMaker();

    // pre-book tick events
    void OnPreBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o) override;

    // post-book tick events
    void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o) override;
    void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o) override;
    void OnPostBookModifyWithPrice(const Timestamp                       event_loop_time,
                                   const BookDataMessageModifyWithPrice *o) override;
    void OnPostBookModifyWithQty(const Timestamp                     event_loop_time,
                                 const BookDataMessageModifyWithQty *o) override;
    void OnPostBookSnapshot(const Timestamp                event_loop_time,
                            const BookDataMessageSnapshot *o) override;

    // trade tick events
    void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o) override;

    // tick event after all event with the same timestamp has already been updated
    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    // taifex order report listener events
    void OnAccepted(const Timestamp event_loop_time, const Symbol *symbol,
                    OrderReportMessageAccepted *o, void *packet) override;
    void OnRejected(const Timestamp event_loop_time, OrderReportMessageRejected *o,
                    void *packet) override;
    void OnCancelled(const Timestamp event_loop_time, const Symbol *symbol,
                     OrderReportMessageCancelled *o, void *packet) override;
    void OnCancelFailed(const Timestamp event_loop_time, OrderReportMessageCancelFailed *o,
                        void *packet) override;
    void OnExecuted(const Timestamp event_loop_time, const Symbol *symbol,
                    OrderReportMessageExecuted *o, void *packet) override;

    void Evaluate(const Timestamp event_loop_time);

    template <typename Side_T>
    void CancelThenSend(const Timestamp &event_loop_time)
    {
        const auto &current_position = std::fabs(position_.GetPosition());
        const auto &side             = Side_T::GetSide();
        if (Side_T::IsInner(last_prediction_, alpha_threshold_[side]))
        {
            const auto &other_side = Side_T::other::GetSide();
            if (outstanding_order_manager_.GetSharesPotential(other_side) > 0)
            {
                auto side_map = outstanding_order_manager_.GetPriceOrderMapFromSide(other_side);
                for (auto &[price, qty_orders_pair] : *side_map)
                {
                    auto &[qty, orders] = qty_orders_pair;
                    for (auto &[orderno, oo] : orders)
                    {
                        if (!oo.last_cancel_time_.is_valid() ||
                            oo.last_cancel_time_ + insert_cancel_duration_ < event_loop_time)
                        {
                            outstanding_order_manager_.InsertCancelOrder(other_side, orderno,
                                                                         price);
                            oo.last_cancel_time_ = event_loop_time;
                        }
                    }
                }
            }

            for (auto &[orderno, cancel_info] : *outstanding_order_manager_.GetCancellingMap())
            {
                if (!cancel_info.last_cancel_time_.is_valid() ||
                    cancel_info.last_cancel_time_ + cancel_wait_duration_ < event_loop_time)
                {
                    Cancel(event_loop_time, cancel_info.side_, IntToOrderNo(orderno).c_str());
                    cancel_info.last_cancel_time_ = event_loop_time;
                }
            }

            if (current_position + outstanding_order_manager_.GetSharesPotential(side) <
                max_position_)
            {
                Send(event_loop_time, side, touch_price_[side], 1);
                SPDLOG_DEBUG("{} shares ask = {}, shares bid = {}", side_string_[side],
                             outstanding_order_manager_.GetSharesPotential(other_side),
                             outstanding_order_manager_.GetSharesPotential(side));
            }
            return;
        }
    }
    void Send(const Timestamp event_loop_time, const BookSide side, const BookPrice price,
              const BookQty qty);
    void Cancel(const Timestamp &event_loop_time, const BookSide side, const char *orderno);

  private:
    const MarketByOrderBook *mobook_;

    const std::string fit_tag_;
    const std::string fit_outcome_;

    const double fee_rate_;
    const double fee_cost_;

    const BookQty max_position_;

    const BookQty   min_order_qty_;
    const BookQty   max_order_qty_;
    const BookPrice alpha_threshold_[AskBid];
    const Duration  expired_duration_;
    const Duration  insert_cancel_duration_;
    const Duration  cancel_wait_duration_;

    ExecutedMsgs            executeds_;
    MetricReporter          metric_reporter_;
    OutstandingOrderManager outstanding_order_manager_;

    const unsigned int tx_account_;
    const char         tx_account_flag_;

    const std::string side_string_[AskBid];

    double last_prediction_;
    double last_mid_price_;
    enum Aggressiveness
    {
        Passive            = 0,
        Aggressive         = 1,
        LastAggressiveness = 2
    };
    BookPrice                   touch_price_[AskBid];
    std::unordered_set<OrderNo> orderno_to_be_cancelled_;
};

class TestMakerEnsemble : public Ensemble
{
  public:
    TestMakerEnsemble(const Symbol *symbol, Strategy *strategy, const nlohmann::json node);
    TestMakerEnsemble(const TestMakerEnsemble &) = delete;
    TestMakerEnsemble &operator=(const TestMakerEnsemble &) = delete;

    ~TestMakerEnsemble();
};

class TestMakerStrategy : public Strategy
{
  public:
    TestMakerStrategy(ObjectManager *object_manager, MultiBookManager *multi_book_manager,
                      MultiCounterManager *multi_counter_manager, Engine *engine,
                      TaifexOrderManagerBase *order_manager = nullptr, size_t config_index = 0);
    TestMakerStrategy(const TestMakerStrategy &) = delete;
    TestMakerStrategy &operator=(const TestMakerStrategy &) = delete;

    ~TestMakerStrategy();
};
}  // namespace alphaone

#endif
