#include "MetricReporter.h"

#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/math/Math.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <filesystem>
#include <numeric>

namespace alphaone
{
static constexpr double ANNUAL{365 * Duration ::from_hour(24.0).to_double()};

void to_json(nlohmann::json &j, const OrderExecutedMetrics &o)
{
    double correct_price{o.executed_message_.Price / o.decimal_converter_};
    j = nlohmann::json{{"order_no", OrderNoToInt(o.executed_message_.OrderNo)},
                       {"price", correct_price},
                       {"qty", o.executed_message_.Qty},
                       {"side", o.executed_message_.Side},
                       {"leaves_qty", o.executed_message_.LeavesQty},
                       {"executed_timestamp", o.executed_timestamp_.to_string()},
                       {"executed_tick", o.executed_tick_}};
}

MetricReporter::MetricReporter(const nlohmann::json &node, const Symbol *symbol, const Book *book,
                               const MultiCounterManager *multi_counter_manager, Engine *engine,
                               const Position *position, nlohmann ::json *custom_json,
                               const size_t &id)
    : symbol_{symbol}
    , book_{book}
    , position_{position}
    , counter_interval_{node.value("counter_interval", "SingleTickInterval")}
    , counter_interval_value_{node.value("counter_interval_value", 256U)}
    , counter_{multi_counter_manager != nullptr
                   ? multi_counter_manager->GetCounter(symbol, counter_interval_)
                   : nullptr}
    , report_interval_{Duration::from_time(node.value("interval", "60s").c_str())}
    , start_time_{node.contains("start_time")
                      ? Timestamp::from_date_time(engine->GetDate(),
                                                  node["start_time"].get<std::string>().c_str())
                      : engine->GetDayStart()}
    , end_time_{node.contains("end_time")
                    ? Timestamp::from_date_time(engine->GetDate(),
                                                node["end_time"].get<std::string>().c_str())
                    : engine->GetDayEnd()}
    , default_json_(nlohmann::json{})
    , mode_{UpdateMode::Time}
    , last_snapshot_timestamp_{Timestamp::invalid()}
    , last_snapshot_asset_{0.0}
    , last_price_{0., 0.}
    , peak_{0.0}
    , drawdown_{0.0}
    , maximum_drawdown_{0.0}
    , observe_count_{0}
    , excess_returns_sum_{0.}
    , square_excess_returns_sum_{0.}
    , positive_observe_count_{0}
    , square_excess_p_returns_sum_{0}
    , writer_{nullptr}
    , done_offset_{0}
    , date_{engine->GetDate()}
    , enable_event_log_{node.value("enable_event_log", false)}
    , last_tick_{0}
    , qty_sent_{0}
{
    if (end_time_ <= start_time_)
    {
        end_time_ += Duration::from_hour(24L);
    }

    if (node.contains("mode"))
    {
        if (node.is_string())
        {
            mode_ = StringToMode(node["mode"].get<std::string>());
        }
        else if (node["mode"].is_number_integer())
        {
            mode_ = static_cast<UpdateMode>(node["mode"].get<int>());
        }
        else
        {
            SPDLOG_ERROR("wrong argument {} for mode", node["mode"].get<std::string>());
            abort();
        }
    }

    switch (mode_)
    {
    case UpdateMode::Time:
        engine->AddPeriodicTimer(start_time_, report_interval_, end_time_,
                                 dynamic_cast<TimerListener *>(this), custom_json);
        break;
    case UpdateMode::Event:
        SPDLOG_WARN("need to manual call timer to update the stats");
        break;
    default:
        SPDLOG_WARN("Undefined mode thus won't trigger");
        break;
    }


    if (node.contains("writer_path"))
    {
        const std::string writer_path{node["writer_path"].get<std::string>()};
        if (!std::filesystem::exists(writer_path))
        {
            std::filesystem::create_directories(writer_path);
        }

        if (node.contains("writer_name"))
        {
            std::string writer_name{date_.to_string() + "." +
                                    node["writer_name"].get<std::string>()};

            if (!engine->IsSimulation())
            {
                writer_name += "." + symbol_->GetRepresentativePid();
                if (start_time_ > Timestamp::from_date_time(date_, "13:45:00.000000000"))
                {
                    writer_name += "_Night";
                }
                if (auto p = writer_name.find_last_of("/"); p != std::string::npos)
                {
                    writer_name.replace(p, 1, "_");
                }
                if (id != 0)
                {
                    char buffer[25];
                    sprintf(buffer, "%02lu", id);
                    writer_name += "." + std::string(buffer);
                }
            }

            writer_ = new JsonWriter(writer_path, writer_name);
        }
        else
        {
            writer_ = new JsonWriter(writer_path);
        }
    }

    all_orders_executed_pnl_over_ticks_.resize(counter_interval_value_, 0.);
    all_orders_executed_pnl_square_over_ticks_.resize(counter_interval_value_, 0.);
    all_orders_executed_pnl_count_over_ticks_.resize(counter_interval_value_, 0);
    sharpe_ratio_by_counter_.resize(counter_interval_value_, 0.);
}

MetricReporter::~MetricReporter()
{
    SPDLOG_INFO("[{}] sharpe_ratio = {}, sortino_ratio = {}, maximum_draw_down = {}", __func__,
                GetSharpeRatio(), GetSortinoRatio(), GetMaximumDrawdown());

    if (!order_executed_pnl_over_ticks_.empty() &&
        spdlog::default_logger()->level() == spdlog::level::debug)
    {
        SPDLOG_DEBUG("Dumping order executed pnl for observed tick");
        SPDLOG_DEBUG("Dumping accumulated order executed pnl for observed tick");
        SPDLOG_DEBUG("{}", fmt::join(all_orders_executed_pnl_over_ticks_, ","));
    }

    if (writer_)
    {
        writer_->InsertByKey("Date", start_time_.to_date().to_string());
        writer_->InsertByKey("Symbol", symbol_->to_string());

        nlohmann::json markouts;
        markouts["Interval"]             = counter_interval_;
        markouts["ProfitOrLossNetValue"] = all_orders_executed_pnl_over_ticks_;
        for (size_t i = 0; i < all_orders_executed_pnl_over_ticks_.size(); ++i)
        {
            sharpe_ratio_by_counter_[i] = GetSharpeRatio(i);
        }
        markouts["SharpeRatio"] = sharpe_ratio_by_counter_;
        writer_->InsertByKey("Markouts", markouts);

        if (position_)
        {
            writer_->InsertByKey("Position", position_->GetJson());
        }

        writer_->InsertByKey("MaximumDrawDown", GetMaximumDrawdown());
        writer_->InsertByKey("FilledRate", GetFilledRate());
        writer_->InsertByKey("ExecutedUnitPnL", GetExecutedUnitPnL());
        writer_->InsertByKey("SentUnitPnL", GetSentUnitPnL());

        if (mode_ == UpdateMode::Time)
        {
            writer_->InsertByKey(
                "SharpeRatio", GetSharpeRatio() * std::sqrt(ANNUAL / report_interval_.to_double()));
            writer_->InsertByKey("SharpeRatioNotAnnual", GetSharpeRatio());
            writer_->InsertByKey("SortinoRatio",
                                 GetSortinoRatio() *
                                     std::sqrt(ANNUAL / report_interval_.to_double()));
            writer_->InsertByKey("SortinoRatioNotAnnual", GetSortinoRatio());
        }
        writer_->InsertByKey("ExcessReturnsSum", excess_returns_sum_);
        writer_->InsertByKey("SquareExcessReturnsSum", square_excess_returns_sum_);
        writer_->InsertByKey("ObserveCount", observe_count_);

        for (const auto &[key, attribute] : extra_attributes_)
        {
            writer_->InsertByKey(key, attribute);
        }

        delete writer_;
        writer_ = nullptr;
    }
}

void MetricReporter::Timer(const Timestamp event_loop_time, const Timestamp call_back_time,
                           void *structure)
{
    Update(event_loop_time, reinterpret_cast<nlohmann::json *>(structure));
}


void MetricReporter::Update(const Timestamp &event_loop_time, nlohmann::json *custom_json)
{
    const double this_snapshot_asset{position_->GetProfitOrLossNetValue()};
    const double this_position{position_->GetPosition()};

    if (book_->IsValid() || this_position == 0)
    {
        peak_ = std::max(peak_, this_snapshot_asset);
    }

    const double elapse_time{(event_loop_time - last_snapshot_timestamp_).to_double()};
    const double excess_return{this_snapshot_asset - last_snapshot_asset_};
    const double annualized_excess_return{excess_return * ANNUAL / elapse_time};

    drawdown_         = std::min(this_snapshot_asset - peak_, 0.0);
    maximum_drawdown_ = std::min(maximum_drawdown_, drawdown_);
    excess_returns_sum_ += annualized_excess_return;
    square_excess_returns_sum_ += annualized_excess_return * annualized_excess_return;
    ++observe_count_;

    if (annualized_excess_return >= 0)
    {
        square_excess_p_returns_sum_ += annualized_excess_return * annualized_excess_return;
        ++positive_observe_count_;
    }

    if (writer_ && enable_event_log_)
    {
        if (book_->IsValid())
        {
            last_price_[Bid] = book_->GetPrice(BID);
            last_price_[Ask] = book_->GetPrice(ASK);
        }

        nlohmann::json snapshot;  // could add more here
        snapshot["Time"]       = event_loop_time.to_string();
        snapshot["TradePrice"] = book_->GetLastTradePrice();
        snapshot["BidPrice"]   = last_price_[Bid];
        snapshot["AskPrice"]   = last_price_[Ask];
        snapshot["Position"]   = position_->GetJson();
        snapshot["Drawdown"]   = drawdown_;
        snapshot["CustomJson"] = (custom_json) ? *custom_json : default_json_;
        if (BRANCH_LIKELY(last_price_[Bid] && last_price_[Ask]))
        {
            writer_->InsertReport(snapshot);
        }
    }

#ifdef AlphaOneDebug
    std::cout << "================================================================\n";
    printf("last_snapshot_asset     : %16.4f\n", last_snapshot_asset_);
    printf("this_snapshot_asset     : %16.4f\n", this_snapshot_asset);
    printf("excess_return           : %16.4f\n", excess_return);
    printf("annualized_excess_return: %16.4f\n", annualized_excess_return);
    printf("elapse_time             : %16.4f\n", elapse_time);
    printf("peak                    : %16.4f\n", peak_);
    printf("drawdown                : %16.4f\n", drawdown_);
    printf("maximum_drawdown        : %16.4f\n", maximum_drawdown_);
    std::cout << "================================================================\n";
#endif

    last_snapshot_timestamp_ = event_loop_time;
    last_snapshot_asset_     = this_snapshot_asset;
}


void MetricReporter::UpdateOrderExecuted(
    const Timestamp &                                              event_loop_time,
    std::vector<std::pair<OrderReportMessageExecuted, Timestamp>> &order_executeds,
    nlohmann::json *                                               custom_json)
{
    if (order_executeds.empty())
    {
        return;
    }

    if (!counter_)
    {
        return;
    }

    const auto &current_tick = static_cast<uint32_t>(counter_->GetCount());
    for (auto it = order_executeds.begin(); it != order_executeds.end(); ++it)
    {
        order_executed_pnl_over_ticks_.push_back({std::move(it->first), std::move(it->second),
                                                  current_tick, symbol_->GetDecimalConverter()});
    }

    if (writer_ && enable_event_log_)
    {
        nlohmann::json snapshot;
        snapshot["Time"]       = event_loop_time.to_string();
        snapshot["TradePrice"] = book_->GetLastTradePrice();
        snapshot["BidPrice"]   = book_->GetPrice(BID);
        snapshot["AskPrice"]   = book_->GetPrice(ASK);
        snapshot["Position"]   = position_->GetJson();
        snapshot["Drawdown"]   = drawdown_;
        snapshot["CustomJson"] = (custom_json) ? *custom_json : default_json_;
        writer_->InsertReport(snapshot);
    }
}

void MetricReporter::UpdateCounterMetrics()
{
    if (!counter_)
    {
        return;
    }

    const auto &current_tick = counter_->GetCount();
    if (last_tick_ == current_tick)
    {
        return;
    }
    last_tick_ = current_tick;
    for (auto it = order_executed_pnl_over_ticks_.begin() + done_offset_;
         it != order_executed_pnl_over_ticks_.end(); ++it)
    {
        const auto &passed_tick = current_tick - it->executed_tick_;
        if (passed_tick > counter_interval_value_ - 1 || it->executed_message_.Price == 0 ||
            it->executed_message_.Qty == 0)
        {
            ++done_offset_;
            continue;
        }
        // We could cache cost in the future if needed
        const auto &pnl = (it->executed_message_.Side == OrderReportSide::Buy)
                              ? (book_->GetMidPrice() -
                                 it->executed_message_.Price / symbol_->GetDecimalConverter()) *
                                        it->executed_message_.Qty * symbol_->GetMultiplier() -
                                    position_->GetTradeCost(BID, it->executed_message_.Price,
                                                            it->executed_message_.Qty)
                              : -(book_->GetMidPrice() -
                                  it->executed_message_.Price / symbol_->GetDecimalConverter()) *
                                        it->executed_message_.Qty * symbol_->GetMultiplier() -
                                    position_->GetTradeCost(ASK, it->executed_message_.Price,
                                                            it->executed_message_.Qty);
        all_orders_executed_pnl_over_ticks_[passed_tick] += pnl;
        all_orders_executed_pnl_square_over_ticks_[passed_tick] += pnl * pnl;
        ++all_orders_executed_pnl_count_over_ticks_[passed_tick];
    }
}

void MetricReporter::UpdateQtySent(const Timestamp &event_loop_time, const BookQty &qty_sent)
{
    qty_sent_ += qty_sent;
}

void MetricReporter::AddAttribute(const std::string &name, const nlohmann::json &attribute)
{
    extra_attributes_.emplace(name, attribute);
}

double MetricReporter::GetSharpeRatio() const
{
    const double excess_return_avg{excess_returns_sum_ / observe_count_};
    const double excess_return_std{std::sqrt(square_excess_returns_sum_ / (observe_count_ - 1) -
                                             excess_return_avg * excess_return_avg)};
    return excess_return_std > 0.0 ? excess_return_avg / excess_return_std : 0.0;
}

double MetricReporter::GetSharpeRatio(const size_t count_index) const
{
    const auto count_size{all_orders_executed_pnl_over_ticks_.size()};
    if (count_index >= count_size)
    {
        SPDLOG_ERROR("index {} >= size {}", count_index, count_size);
        return 0.;
    }
    const auto   observe_count{all_orders_executed_pnl_count_over_ticks_[count_index]};
    const double avg{all_orders_executed_pnl_over_ticks_[count_index] / observe_count};
    const double std{std::sqrt(
        all_orders_executed_pnl_square_over_ticks_[count_index] / (observe_count - 1) - avg * avg)};
    return std > 0. ? avg / std : 0.;
}

double MetricReporter::GetSortinoRatio() const
{
    const double excess_return_avg{excess_returns_sum_ / observe_count_};
    const double excess_return_std{
        std::sqrt(square_excess_p_returns_sum_ / (positive_observe_count_ - 1) -
                  excess_return_avg * excess_return_avg)};
    return excess_return_std > 0.0 ? excess_return_avg / excess_return_std : 0.0;
}

double MetricReporter::GetMaximumDrawdown() const
{
    return maximum_drawdown_;
}

double MetricReporter::GetFilledRate() const
{
    return qty_sent_ ? position_->GetTurnOver() / qty_sent_ : 0;
}

double MetricReporter::GetQtySent() const
{
    return qty_sent_;
}

double MetricReporter::GetExecutedUnitPnL() const
{
    const auto &to = position_->GetTurnOver();
    return to ? position_->GetProfitOrLossNetValue() / to : 0;
}

double MetricReporter::GetSentUnitPnL() const
{
    return qty_sent_ ? position_->GetProfitOrLossNetValue() / qty_sent_ : 0;
}

void MetricReporter::SetEnableEventLog(const bool enable_event_log)
{
    enable_event_log_ = enable_event_log;
}
}  // namespace alphaone
