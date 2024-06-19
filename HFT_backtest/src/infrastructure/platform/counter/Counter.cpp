#include "Counter.h"

namespace alphaone
{
Counter::Counter(const Book *book, MultiBookManager *multi_book_manager)
    : symbol_{book->GetSymbol()}
    , book_{book}
    , multi_book_manager_{multi_book_manager}
    , warmed_up_{false}
    , last_update_timestamp_{Timestamp::invalid()}
    , count_{0}
{
    elements_.reserve(8);
}

void Counter::EmitSnapshot(const Timestamp event_loop_time)
{
    for (auto &listener : snapshot_listeners_)
    {
        listener->OnSnapshot(event_loop_time);
    }
}

void Counter::EmitSignal(const Timestamp event_loop_time)
{
    for (auto &listener : signal_listeners_)
    {
        listener->OnSignal(event_loop_time);
    }
}

bool Counter::IsPrepared() const
{
    return book_->IsValid();
}

bool Counter::IsWarmedUp() const
{
    return warmed_up_;
}

void Counter::SetElements()
{
    elements_.emplace_back(Name());
    if (symbol_ != nullptr)
        elements_.emplace_back(symbol_->GetRepresentativePid());
}

const Book *Counter::GetBook() const
{
    return book_;
}

size_t Counter::GetCount() const
{
    return count_;
}

Timestamp Counter::GetLastUpdateTime() const
{
    return last_update_timestamp_;
}

std::string Counter::Name() const
{
    return "Counter";
}

std::string Counter::ToString() const
{
    return fmt::format("{}", fmt::join(elements_, "_"));
}

void Counter::Dump() const
{
    const auto &str = ToString();
    SPDLOG_INFO("{} [{:>12}] [{}] ", GetLastUpdateTime(), GetCount(), str);
}

void Counter::AddSignalCounterListener(CounterListener *listener) const
{
    if (std::find(signal_listeners_.begin(), signal_listeners_.end(), listener) ==
        signal_listeners_.end())
    {
        signal_listeners_.push_back(listener);
    }
}

void Counter::AddSnapshotCounterListener(CounterListener *listener) const
{
    if (std::find(snapshot_listeners_.begin(), snapshot_listeners_.end(), listener) ==
        snapshot_listeners_.end())
    {
        snapshot_listeners_.push_back(listener);
    }
}

void Counter::SubscribeBook(const Book *book)
{
    if (IsWarmedUp())
    {
        if (multi_book_manager_ != nullptr)
        {
            multi_book_manager_->AddPreBookListener(book, this);
            multi_book_manager_->AddPostBookListener(book, this);
        }
        else
        {
            SPDLOG_ERROR(
                "[Counter::{}] trying to subscribe book when multi_book_manager is nullptr",
                __func__);
            abort();
        }
    }
    else
    {
        SPDLOG_ERROR("[Counter::{}] trying to subscribe book before being warmed up", __func__);
        abort();
    }
}

void Counter::SubscribeSymbol(const Symbol *symbol)
{
    if (IsWarmedUp())
    {
        if (multi_book_manager_ != nullptr)
        {
            multi_book_manager_->AddPreBookListener(symbol, this);
            multi_book_manager_->AddPostBookListener(symbol, this);
        }
        else
        {
            SPDLOG_ERROR(
                "[Counter::{}] trying to subscribe symbol when multi_book_manager is nullptr",
                __func__);
            abort();
        }
    }
    else
    {
        SPDLOG_ERROR("[Counter::{}] trying to subscribe symbol before being warmed up", __func__);
        abort();
    }
}

void Counter::RegisterSymbol(const Symbol *symbol, const DataSourceType &type)
{
    if (multi_book_manager_ != nullptr)
    {
        multi_book_manager_->AddSymbolToUniverse(symbol, type);
    }
    else
    {
        SPDLOG_ERROR("[Counter::{}] trying to register symbol when multi_book_manager is nullptr",
                     __func__);
        abort();
    }
}
}  // namespace alphaone
