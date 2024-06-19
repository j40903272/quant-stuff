#ifndef _METRICREPORTER_H_
#define _METRICREPORTER_H_

#include "counter/MultiCounterManager.h"
#include "infrastructure/base/TimerListener.h"
#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/common/position/Position.h"
#include "infrastructure/common/util/Order.h"
#include "infrastructure/platform/engine/Engine.h"
#include "infrastructure/platform/writer/JsonWriter.h"

namespace alphaone
{
struct OrderExecutedMetrics
{
    OrderReportMessageExecuted executed_message_;
    Timestamp                  executed_timestamp_;
    uint32_t                   executed_tick_;
    double                     decimal_converter_;
};

void to_json(nlohmann::json &j, const OrderExecutedMetrics &o);

class MetricReporter : public TimerListener
{
  public:
    MetricReporter(const nlohmann::json &node, const Symbol *symbol, const Book *book,
                   const MultiCounterManager *multi_counter_manager, Engine *engine,
                   const Position *position, nlohmann ::json *custom_json = nullptr,
                   const size_t &id = 0);

    virtual ~MetricReporter();

    void Timer(const Timestamp event_loop_time, const Timestamp call_back_time, void *structure);

    void Update(const Timestamp &event_loop_time, nlohmann::json *custom_json = nullptr);
    void UpdateOrderExecuted(
        const Timestamp &                                              event_loop_time,
        std::vector<std::pair<OrderReportMessageExecuted, Timestamp>> &order_executeds,
        nlohmann::json *                                               custom_json = nullptr);
    void UpdateCounterMetrics();
    void UpdateQtySent(const Timestamp &event_loop_time, const BookQty &qty_sent);

    double        GetSharpeRatio() const;
    double        GetSharpeRatio(size_t count_index) const;
    double        GetSortinoRatio() const;
    double        GetMaximumDrawdown() const;
    double        GetFilledRate() const;
    double        GetQtySent() const;
    inline double GetExecutedUnitPnL() const;
    inline double GetSentUnitPnL() const;

    void AddAttribute(const std::string &name, const nlohmann::json &attribute);
    void SetEnableEventLog(const bool enable_event_log);

  private:
    enum UpdateMode
    {
        Time      = 0,
        Event     = 1,
        Undefined = 2,
    };

    UpdateMode StringToMode(const std::string &str)
    {
        if (str.find("Time") != std::string::npos)
        {
            return UpdateMode::Time;
        }
        else if (str.find("Event") != std::string::npos)
        {
            return UpdateMode::Event;
        }
        else
        {
            return UpdateMode::Undefined;
        }
    }

    const Symbol *       symbol_;
    const Book *         book_;
    const Position *     position_;
    const std::string    counter_interval_;
    const uint32_t       counter_interval_value_;
    const Counter *      counter_;
    const Duration       report_interval_;
    const Timestamp      start_time_;
    Timestamp            end_time_;
    const nlohmann::json default_json_;
    UpdateMode           mode_;
    Timestamp            last_snapshot_timestamp_;
    double               last_snapshot_asset_;
    BookPrice            last_price_[AskBid];
    double               peak_;
    double               drawdown_;
    double               maximum_drawdown_;  // negative
    int                  observe_count_;
    double               excess_returns_sum_;
    double               square_excess_returns_sum_;
    int                  positive_observe_count_;
    double               square_excess_p_returns_sum_;
    JsonWriter *         writer_;

    std::vector<OrderExecutedMetrics> order_executed_pnl_over_ticks_;

    std::vector<double> all_orders_executed_pnl_over_ticks_;
    std::vector<double> all_orders_executed_pnl_square_over_ticks_;
    std::vector<int>    all_orders_executed_pnl_count_over_ticks_;
    std::vector<double> sharpe_ratio_by_counter_;
    size_t              done_offset_;
    Date                date_;
    bool                enable_event_log_;
    uint32_t            last_tick_;
    BookQty             qty_sent_;

    std::map<std::string, nlohmann::json> extra_attributes_;
};
}  // namespace alphaone

#endif
