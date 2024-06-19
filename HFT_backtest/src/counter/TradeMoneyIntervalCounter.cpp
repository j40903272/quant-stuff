#include "TradeMoneyIntervalCounter.h"

namespace alphaone
{
TradeMoneyIntervalCounter::TradeMoneyIntervalCounter(const ObjectManager * object_manager,
                                                     MultiBookManager *    multi_book_manager,
                                                     const nlohmann::json &spec)
    : Counter{multi_book_manager->GetBook(
                  object_manager->GetSymbolManager()->GetSymbolByString(spec.value("symbol", ""))),
              multi_book_manager}
    , symbol_{book_->GetSymbol()}
    , accuracy_{spec.value("accuracy", 100000.0)}
    , threshold_{ParseThreshold(object_manager->GetReferenceData()
                                    ->GetSymbolInfoFromSymbol(symbol_)
                                    .GetOhlcvs()) /
                 accuracy_}
    , packet_trade_money_{0.0}
    , this_trade_money_{0.0}
    , last_trade_money_{0.0}
{
    if (accuracy_ < 1.0)
    {
        throw std::invalid_argument(
            fmt::format("accuracy={} should be no less than 1.0", accuracy_));
    }
    SetElements();
}

TradeMoneyIntervalCounter::~TradeMoneyIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void TradeMoneyIntervalCounter::OnTrade(const Timestamp             event_loop_time,
                                        const BookDataMessageTrade *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    packet_trade_money_ += o->GetTradeQty() * o->GetTradePrice();
}

void TradeMoneyIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                            const BookDataMessagePacketEnd *o)
{
    this_trade_money_ += packet_trade_money_;

    if (const size_t tick{
            static_cast<size_t>(std::floor((this_trade_money_ - last_trade_money_) / threshold_))};
        tick > 0)
    {
        last_update_timestamp_ = event_loop_time;
        last_trade_money_ += tick * threshold_;
        count_ += tick;
    }

    packet_trade_money_ = 0.0;
}

void TradeMoneyIntervalCounter::SetElements()
{
    elements_.emplace_back(Name());
    elements_.emplace_back(to_succinct_string(accuracy_));
    if (symbol_ != nullptr)
    {
        elements_.emplace_back(symbol_->GetRepresentativePid());
    }
}

BookQty TradeMoneyIntervalCounter::GetTotalTradeQty() const
{
    return this_trade_money_;
}

std::string TradeMoneyIntervalCounter::Name() const
{
    return "TradeMoneyInterval";
}

void TradeMoneyIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_trade_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] last_total_trade_qty: {}", str, last_trade_money_);
    SPDLOG_INFO("[{}] trade_qty_count: {}", str, count_);
    SPDLOG_INFO("[{}] total_trade_qty: {}", str, this_trade_money_);
}

double TradeMoneyIntervalCounter::ParseThreshold(
    const std::vector<Ohlcv<BookPrice, BookQty>> &ohlcvs) const
{
    double result{0.0};
    for (const auto &item : ohlcvs)
    {
        result += item.volume_ * (0.5 * (item.open_ + item.close_)) / ohlcvs.size();
    }
    return result;
}
}  // namespace alphaone
