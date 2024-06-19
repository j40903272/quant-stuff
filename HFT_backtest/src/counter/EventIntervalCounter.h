#ifndef _EVENTINTERVALCOUNTER_H_
#define _EVENTINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class EventIntervalCounter : public Counter
{
  public:
    EventIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    EventIntervalCounter(const EventIntervalCounter &) = delete;
    EventIntervalCounter &operator=(const EventIntervalCounter &) = delete;
    EventIntervalCounter(EventIntervalCounter &&)                 = delete;
    EventIntervalCounter &operator=(EventIntervalCounter &&) = delete;

    ~EventIntervalCounter();

    void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o) override;
    void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o) override;
    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
    size_t fragment_events_;
};
}  // namespace alphaone

#endif
