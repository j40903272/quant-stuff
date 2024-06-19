#include "TouchIntervalCounter.h"

namespace alphaone
{
TouchIntervalCounter::TouchIntervalCounter(const Book *book, MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}
{
    SetElements();
}

TouchIntervalCounter::~TouchIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void TouchIntervalCounter::OnPostBookAdd(const Timestamp           event_loop_time,
                                         const BookDataMessageAdd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    if (o->GetMarketByOrderPrice() == book_->GetPrice(o->GetMarketByOrderSide()))
    {
        is_touch_ = true;
    }
}

void TouchIntervalCounter::OnPostBookDelete(const Timestamp              event_loop_time,
                                            const BookDataMessageDelete *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    if (o->GetMarketByOrderPrice() == book_->GetPrice(o->GetMarketByOrderSide()))
    {
        is_touch_ = true;
    }
}

void TouchIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                       const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    if (is_touch_)
    {
        last_update_timestamp_ = event_loop_time;
        count_ += 1;

        is_touch_ = false;
    }
}

std::string TouchIntervalCounter::Name() const
{
    return "TouchInterval";
}

void TouchIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_touch_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] touches: {}", str, count_);
}
}  // namespace alphaone
