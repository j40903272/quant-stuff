#include "MultiCounterManager.h"

namespace alphaone
{
MultiCounterManager::MultiCounterManager(const ObjectManager *object_manager,
                                         MultiBookManager *multi_book_manager, Engine *engine)
    : object_manager_{object_manager}
    , symbol_manager_{object_manager->GetSymbolManager()}
    , multi_book_manager_{multi_book_manager}
    , engine_{engine}
{
#ifdef AlphaOneMultiCounterManagerDebug
    for (const auto &symbol : multi_book_manager_->GetUniverse())
    {
        AddCounter(symbol, "AddInterval");
        AddCounter(symbol, "AxisInterval");
        AddCounter(symbol, "DeleteInterval");
        AddCounter(symbol, "DoubleTickInterval");
        AddCounter(symbol, "HalfTickInterval");
        AddCounter(symbol, "MessageInterval");
        AddCounter(symbol, "SingleTickInterval");
        AddCounter(symbol, "TimeInterval");
        AddCounter(symbol, "TimeInterval_2");
        AddCounter(symbol, "TimeInterval_2.0");
        AddCounter(symbol, "TimeInterval_5");
        AddCounter(symbol, "TimeInterval_5.0");
        AddCounter(symbol, "TimeInterval_0.5");
        AddCounter(symbol, "TouchInterval");
        AddCounter(symbol, "TradeInterval");
        AddCounter(symbol, "TradeQtyInterval");
        AddCounter(symbol, "TradeQtyInterval_2");
        AddCounter(symbol, "TradeQtyInterval_5");
        AddCounter(symbol, "TradeQtyInterval_0.5");
        AddCounter(symbol, "TradeThroughInterval");
        AddCounter(symbol, "WeightedTickInterval");
    }
#endif
}

MultiCounterManager::~MultiCounterManager()
{
    for (auto &[counter_name, counter] : registered_counters_)
    {
        if (counter != nullptr)
        {
            delete counter;
            counter = nullptr;
        }
    }
}

void MultiCounterManager::OnPreBookDelete(const Timestamp              event_loop_time,
                                          const BookDataMessageDelete *o)
{
    EmitSnapshot(event_loop_time);
    for (auto &listener : registered_listeners_)
    {
        listener->OnPreBookDelete(event_loop_time, o);
    }
    EmitSignal(event_loop_time);
}

void MultiCounterManager::OnPostBookAdd(const Timestamp           event_loop_time,
                                        const BookDataMessageAdd *o)
{
    EmitSnapshot(event_loop_time);
    for (auto &listener : registered_listeners_)
    {
        listener->OnPostBookAdd(event_loop_time, o);
    }
    EmitSignal(event_loop_time);
}

void MultiCounterManager::OnPostBookDelete(const Timestamp              event_loop_time,
                                           const BookDataMessageDelete *o)
{
    EmitSnapshot(event_loop_time);
    for (auto &listener : registered_listeners_)
    {
        listener->OnPostBookDelete(event_loop_time, o);
    }
    EmitSignal(event_loop_time);
}

void MultiCounterManager::OnPostBookModifyWithPrice(const Timestamp event_loop_time,
                                                    const BookDataMessageModifyWithPrice *o)
{
    EmitSnapshot(event_loop_time);
    for (auto &listener : registered_listeners_)
    {
        listener->OnPostBookModifyWithPrice(event_loop_time, o);
    }
    EmitSignal(event_loop_time);
}

void MultiCounterManager::OnPostBookModifyWithQty(const Timestamp event_loop_time,
                                                  const BookDataMessageModifyWithQty *o)
{
    EmitSnapshot(event_loop_time);
    for (auto &listener : registered_listeners_)
    {
        listener->OnPostBookModifyWithQty(event_loop_time, o);
    }
    EmitSignal(event_loop_time);
}

void MultiCounterManager::OnPostBookSnapshot(const Timestamp                event_loop_time,
                                             const BookDataMessageSnapshot *o)
{
    EmitSnapshot(event_loop_time);
    for (auto &listener : registered_listeners_)
    {
        listener->OnPostBookSnapshot(event_loop_time, o);
    }
    EmitSignal(event_loop_time);
}

void MultiCounterManager::OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o)
{
    EmitSnapshot(event_loop_time);
    for (auto &listener : registered_listeners_)
    {
        listener->OnTrade(event_loop_time, o);
    }
    EmitSignal(event_loop_time);
}

void MultiCounterManager::OnPacketEnd(const Timestamp                 event_loop_time,
                                      const BookDataMessagePacketEnd *o)
{
    EmitSnapshot(event_loop_time);
    for (auto &listener : registered_listeners_)
    {
        listener->OnPacketEnd(event_loop_time, o);
    }
    EmitSignal(event_loop_time);
}

void MultiCounterManager::OnSparseStop(const Timestamp &                event_loop_time,
                                       const BookDataMessageSparseStop *o)
{
    EmitSnapshot(event_loop_time);
    for (auto &listener : registered_listeners_)
    {
        listener->OnSparseStop(event_loop_time, o);
    }
    EmitSignal(event_loop_time);
}

const Counter *MultiCounterManager::GetCounter(const nlohmann::json &spec) const
{
    const std::string &interval{spec["interval"].get<std::string>()};

    Counter *counter{CounterFactory::RetrieveCounterFromCounterSpec(
        object_manager_, multi_book_manager_, engine_, interval, spec)};

    if (counter != nullptr)
    {
        auto [it, is_emplace_success] = registered_counters_.emplace(counter->ToString(), counter);
        if (is_emplace_success)
        {
            SPDLOG_INFO("[MultiCounterManager::{}] construct {}", __func__, spec.dump());
            counter->WarmUp();
        }
        else
        {
            delete counter;
        }

        return it->second;
    }

    SPDLOG_WARN("[MultiCounterManager::{}] cannot resolve {}", __func__, spec.dump());
    return nullptr;
}

const Counter *MultiCounterManager::GetCounter(const std::string &   interval,
                                               const nlohmann::json &spec) const
{
    nlohmann::json json(spec);
    json["interval"] = interval;
    return GetCounter(json);
}

const Counter *MultiCounterManager::GetCounter(const Symbol *symbol, const std::string &interval,
                                               const nlohmann::json &spec) const
{
    nlohmann::json json(spec);
    json["symbol"] = symbol->to_string();
    return GetCounter(interval, json);
}

void MultiCounterManager::AddCounterBookDataListener(BookDataListener *listener)
{
    registered_listeners_.emplace_back(listener);
}

void MultiCounterManager::EmitSnapshot(const Timestamp event_loop_time)
{
    for (auto &[counter_name, counter] : registered_counters_)
    {
        counter->EmitSnapshot(event_loop_time);
    }
}

void MultiCounterManager::EmitSignal(const Timestamp event_loop_time)
{
    for (auto &[counter_name, counter] : registered_counters_)
    {
        counter->EmitSignal(event_loop_time);
    }
}
}  // namespace alphaone
