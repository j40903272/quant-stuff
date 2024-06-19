#ifndef _COUNTER_H_
#define _COUNTER_H_

#include "infrastructure/base/Book.h"
#include "infrastructure/base/BookDataListener.h"
#include "infrastructure/base/CounterListener.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/util/Logger.h"
#include "infrastructure/platform/manager/MultiBookManager.h"

namespace alphaone
{
class Counter : public BookDataListener
{
  public:
    Counter(const Book *book, MultiBookManager *multi_book_manager);
    Counter(const Counter &) = delete;
    Counter &operator=(const Counter &) = delete;
    Counter(Counter &&)                 = delete;
    Counter &operator=(Counter &&) = delete;

    virtual ~Counter() = default;

    virtual void OnPreBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
    {
    }
    virtual void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o)
    {
    }
    virtual void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
    {
    }
    virtual void OnPostBookModifyWithPrice(const Timestamp                       event_loop_time,
                                           const BookDataMessageModifyWithPrice *o)
    {
    }
    virtual void OnPostBookModifyWithQty(const Timestamp                     event_loop_time,
                                         const BookDataMessageModifyWithQty *o)
    {
    }
    virtual void OnPostBookSnapshot(const Timestamp                event_loop_time,
                                    const BookDataMessageSnapshot *o)
    {
    }
    virtual void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o)
    {
    }
    virtual void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o)
    {
    }
    virtual void OnSparseStop(const Timestamp &event_loop_time, const BookDataMessageSparseStop *o)
    {
    }
    virtual void OnTPrice(const Timestamp &event_loop_time, const MarketDataMessage *t)
    {
    }

    virtual void EmitSnapshot(const Timestamp event_loop_time);
    virtual void EmitSignal(const Timestamp event_loop_time);

    virtual inline void WarmUp();

    bool IsPrepared() const;
    bool IsWarmedUp() const;

    const Book *        GetBook() const;
    virtual size_t      GetCount() const;
    virtual Timestamp   GetLastUpdateTime() const;
    virtual std::string Name() const;
    virtual std::string ToString() const;
    virtual void        Dump() const;

    void AddSignalCounterListener(CounterListener *listener) const;
    void AddSnapshotCounterListener(CounterListener *listener) const;

  protected:
    virtual void SetElements();

    void SubscribeBook(const Book *book);
    void SubscribeSymbol(const Symbol *symbol);
    void RegisterSymbol(const Symbol *        symbol,
                        const DataSourceType &type = DataSourceType::MarketByOrder);

    // required basic components
    const Symbol *    symbol_;
    const Book *      book_;
    MultiBookManager *multi_book_manager_;

    // listeners
    mutable std::vector<CounterListener *> signal_listeners_;
    mutable std::vector<CounterListener *> snapshot_listeners_;

    // base atttributes
    bool      warmed_up_;
    Timestamp last_update_timestamp_;
    size_t    count_;

    // for ToString
    std::vector<std::string> elements_;
};

inline void Counter::WarmUp()
{
    warmed_up_ = true;
    SubscribeBook(book_);
}
}  // namespace alphaone

#endif
