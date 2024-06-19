#ifndef _MULTICOUNTERMANAGER_H_
#define _MULTICOUNTERMANAGER_H_

#include "counter/CounterFactory.h"
#include "infrastructure/platform/counter/Counter.h"
#include "infrastructure/platform/manager/MultiBookManager.h"

namespace alphaone
{
class MultiCounterManager : public BookDataListener
{
  public:
    MultiCounterManager(const ObjectManager *object_manager, MultiBookManager *multi_book_manager,
                        Engine *engine);
    MultiCounterManager(const MultiCounterManager &) = delete;
    MultiCounterManager &operator=(const MultiCounterManager &) = delete;

    ~MultiCounterManager();

    void OnPreBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o);
    void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o);
    void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o);
    void OnPostBookModifyWithPrice(const Timestamp                       event_loop_time,
                                   const BookDataMessageModifyWithPrice *o);
    void OnPostBookModifyWithQty(const Timestamp                     event_loop_time,
                                 const BookDataMessageModifyWithQty *o);
    void OnPostBookSnapshot(const Timestamp event_loop_time, const BookDataMessageSnapshot *o);
    void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o);
    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o);
    void OnSparseStop(const Timestamp &event_loop_time, const BookDataMessageSparseStop *o);

    const Counter *GetCounter(const nlohmann::json &spec) const;
    const Counter *GetCounter(const std::string &interval, const nlohmann::json &spec) const;
    const Counter *GetCounter(const Symbol *symbol, const std::string &interval,
                              const nlohmann::json &spec = nlohmann::json::value_t::object) const;

    void AddCounterBookDataListener(BookDataListener *collator);

  private:
    void EmitSnapshot(const Timestamp event_loop_time);
    void EmitSignal(const Timestamp event_loop_time);

    const ObjectManager *                    object_manager_;
    const SymbolManager *                    symbol_manager_;
    MultiBookManager *                       multi_book_manager_;
    Engine *                                 engine_;
    std::vector<BookDataListener *>          registered_listeners_;
    mutable std::map<std::string, Counter *> registered_counters_;
};
}  // namespace alphaone

#endif
