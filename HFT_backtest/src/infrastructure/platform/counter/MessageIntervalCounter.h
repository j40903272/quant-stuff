#ifndef _MESSAGEINTERVALCOUNTER_H_
#define _MESSAGEINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class MessageIntervalCounter : public Counter
{
  public:
    MessageIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    MessageIntervalCounter(const MessageIntervalCounter &) = delete;
    MessageIntervalCounter &operator=(const MessageIntervalCounter &) = delete;
    MessageIntervalCounter(MessageIntervalCounter &&)                 = delete;
    MessageIntervalCounter &operator=(MessageIntervalCounter &&) = delete;

    ~MessageIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
};
}  // namespace alphaone

#endif
