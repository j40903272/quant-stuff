#include "EventIntervalCounter.h"

namespace alphaone
{
EventIntervalCounter::EventIntervalCounter(const Book *book, MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}, fragment_events_{0}
{
    SetElements();
}

EventIntervalCounter::~EventIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void EventIntervalCounter::OnPostBookAdd(const Timestamp           event_loop_time,
                                         const BookDataMessageAdd *o)
{
    if (BRANCH_UNLIKELY(!IsPrepared()))
    {
        return;
    }
    last_update_timestamp_ = event_loop_time;
    ++fragment_events_;
}

void EventIntervalCounter::OnPostBookDelete(const Timestamp              event_loop_time,
                                            const BookDataMessageDelete *o)
{
    if (BRANCH_UNLIKELY(!IsPrepared()))
    {
        return;
    }
    last_update_timestamp_ = event_loop_time;
    ++fragment_events_;
}

void EventIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                       const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }
    count_ += fragment_events_;
    fragment_events_ = 0;
}

std::string EventIntervalCounter::Name() const
{
    return "EventInterval";
}

void EventIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_event_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] events: {}", str, count_);
}
}  // namespace alphaone
