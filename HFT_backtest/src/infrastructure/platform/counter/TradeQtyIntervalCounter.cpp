#include "TradeQtyIntervalCounter.h"

namespace alphaone
{
TradeQtyIntervalCounter::TradeQtyIntervalCounter(const Book *          book,
                                                 MultiBookManager *    multi_book_manager,
                                                 const nlohmann::json &spec)
    : Counter{book, multi_book_manager}
    , threshold_{spec.value("lots", 1.0)}
    , total_trade_qty_{0.0}
    , last_total_trade_qty_{0.0}
{
    SetElements();
}

TradeQtyIntervalCounter::~TradeQtyIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void TradeQtyIntervalCounter::OnTrade(const Timestamp             event_loop_time,
                                      const BookDataMessageTrade *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    total_trade_qty_ += o->GetTradeQty();

    if (const size_t tick{static_cast<size_t>(
            std::floor((total_trade_qty_ - last_total_trade_qty_) / threshold_))};
        tick > 0)
    {
        last_update_timestamp_ = event_loop_time;
        last_total_trade_qty_ += tick * threshold_;
        count_ += tick;
    }
}

void TradeQtyIntervalCounter::SetElements()
{
    elements_.emplace_back(Name());
    if (threshold_ != 1.0)
        elements_.emplace_back(to_succinct_string(threshold_));
    if (symbol_ != nullptr)
        elements_.emplace_back(symbol_->GetRepresentativePid());
}

BookQty TradeQtyIntervalCounter::GetTotalTradeQty() const
{
    return total_trade_qty_;
}

std::string TradeQtyIntervalCounter::Name() const
{
    return "TradeQtyInterval";
}

void TradeQtyIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_trade_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] last_total_trade_qty: {}", str, last_total_trade_qty_);
    SPDLOG_INFO("[{}] trade_qty_count: {}", str, count_);
    SPDLOG_INFO("[{}] total_trade_qty: {}", str, total_trade_qty_);
}
}  // namespace alphaone
