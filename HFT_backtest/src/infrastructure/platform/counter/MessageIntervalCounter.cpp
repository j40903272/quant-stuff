#include "MessageIntervalCounter.h"

namespace alphaone
{
MessageIntervalCounter::MessageIntervalCounter(const Book *      book,
                                               MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}
{
    SetElements();
}

MessageIntervalCounter::~MessageIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void MessageIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                         const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    last_update_timestamp_ = event_loop_time;
    count_ += 1;
}

std::string MessageIntervalCounter::Name() const
{
    return "MessageInterval";
}

void MessageIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_message_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] messages: {}", str, count_);
}
}  // namespace alphaone
