#ifndef _TOUCHINTERVALCOUNTER_H_
#define _TOUCHINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class TouchIntervalCounter : public Counter
{
  public:
    TouchIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    TouchIntervalCounter(const TouchIntervalCounter &) = delete;
    TouchIntervalCounter &operator=(const TouchIntervalCounter &) = delete;
    TouchIntervalCounter(TouchIntervalCounter &&)                 = delete;
    TouchIntervalCounter &operator=(TouchIntervalCounter &&) = delete;

    ~TouchIntervalCounter();

    void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o) override;
    void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o) override;
    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
    bool is_touch_;
};
}  // namespace alphaone

#endif
