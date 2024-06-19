#include "DeleteIntervalCounter.h"

namespace alphaone
{
DeleteIntervalCounter::DeleteIntervalCounter(const Book *book, MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}
{
    SetElements();
}

DeleteIntervalCounter::~DeleteIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void DeleteIntervalCounter::OnPostBookDelete(const Timestamp              event_loop_time,
                                             const BookDataMessageDelete *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    last_update_timestamp_ = event_loop_time;
    count_ += 1;
}

std::string DeleteIntervalCounter::Name() const
{
    return "DeleteInterval";
}

void DeleteIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_delete_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] deletes: {}", str, count_);
}
}  // namespace alphaone
