#include "AddIntervalCounter.h"

namespace alphaone
{
AddIntervalCounter::AddIntervalCounter(const Book *book, MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}
{
    SetElements();
}

AddIntervalCounter::~AddIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void AddIntervalCounter::OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    // AddCounter
    last_update_timestamp_ = event_loop_time;
    count_ += 1;
}

std::string AddIntervalCounter::Name() const
{
    return "AddInterval";
}

void AddIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_add_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] adds: {}", str, count_);
}
}  // namespace alphaone
