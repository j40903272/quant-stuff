#ifndef BOOKDATALISTENER_H
#define BOOKDATALISTENER_H

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/message/BookDataMessage.h"
#include "infrastructure/common/symbol/Symbol.h"

namespace alphaone
{
class Book;

/* -------------------------------------------------------------------------------------------------
BookDataListener is an abstract interface represents the interface between connectivity code and
strategy code. Anything from connectivity code should be handled properly before
MarketDataListeners. Book is designed to be a MarketDataListener. Book will take things from
MarketDataSources, handle information in its own way, and then call BookDataListeners back to
trigger events
------------------------------------------------------------------------------------------------- */
class BookDataListener
{
  public:
    BookDataListener()                         = default;
    BookDataListener(const BookDataListener &) = delete;
    BookDataListener &operator=(const BookDataListener &) = delete;

    virtual ~BookDataListener() = default;

    // pre-book tick events
    virtual void OnPreBookDelete(const Timestamp              event_loop_time,
                                 const BookDataMessageDelete *o) = 0;

    // post-book tick events
    virtual void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o) = 0;
    virtual void OnPostBookDelete(const Timestamp              event_loop_time,
                                  const BookDataMessageDelete *o)                            = 0;
    virtual void OnPostBookModifyWithPrice(const Timestamp                       event_loop_time,
                                           const BookDataMessageModifyWithPrice *o)          = 0;
    virtual void OnPostBookModifyWithQty(const Timestamp                     event_loop_time,
                                         const BookDataMessageModifyWithQty *o)              = 0;
    virtual void OnPostBookSnapshot(const Timestamp                event_loop_time,
                                    const BookDataMessageSnapshot *o)                        = 0;

    // trade tick events
    virtual void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o) = 0;

    // tick event after all event with the same timestamp has already been updated
    virtual void OnPacketEnd(const Timestamp                 event_loop_time,
                             const BookDataMessagePacketEnd *o) = 0;

    // warning events
    virtual void OnSparseStop(const Timestamp &                event_loop_time,
                              const BookDataMessageSparseStop *o) = 0;
};

}  // namespace alphaone
#endif
