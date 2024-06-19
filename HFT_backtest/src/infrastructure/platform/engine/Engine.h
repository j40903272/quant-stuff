#ifndef _ENGINE_H_
#define _ENGINE_H_

#include "infrastructure/base/CommandRelay.h"
#include "infrastructure/base/MarketDataListener.h"
#include "infrastructure/base/TaifexOrderManagerBase.h"
#include "infrastructure/base/TimerListener.h"
#include "infrastructure/base/TimerSource.h"
#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/nats/NATSConnection.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/twone/ringbuffer/RingBuffer.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/PacketLog.h"
#include "infrastructure/platform/datasource/MarketDataSource.h"
#include "infrastructure/platform/manager/MarketDataProviderManager.h"
#include "infrastructure/platform/manager/MulticastReceiverManager.h"
#include "infrastructure/platform/manager/RingBufferManager.h"

#include <functional>
#include <thread>

#define TIMER_SOURCES_WITH_MULTIMAP

namespace alphaone
{

class Engine
{
  public:
    Engine(const ObjectManager *object_manager, const Date &date,
           EngineEventLoopType event_loop_type);
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    ~Engine();

    bool IsSimulation() const;

    void AddMarketDataListener(const Symbol *symbol, MarketDataListener *listener);

    // add a one-time timer source (shortcut code)
    void AddOneTimeTimer(const Timestamp timestamp_scheduled, TimerListener *listener, void *data);
    template <typename F, typename P>
    void AddOneTimeTimer(const Timestamp timestamp_scheduled, F function_callback,
                         P pointer_callback_class, void *structure);
    template <typename F, typename P>
    void AddOneTimeTimer(const Timestamp timestamp_scheduled, F function_callback,
                         std::vector<P> pointer_callback_classes, void *data);
    void AddPeriodicTimer(const Timestamp timestamp_scheduled, const Duration &duration_periodicity,
                          TimerListener *listener, void *data);
    void AddPeriodicTimer(const Timestamp timestamp_start, const Duration &duration_periodicity,
                          const Timestamp timestamp_end, TimerListener *listener, void *data);
    template <typename F, typename P>
    void AddPeriodicTimer(const Timestamp timestamp_scheduled, const Duration &duration_periodicity,
                          F function_callback, P pointer_callback_class, void *structure);
    template <typename F, typename P>
    void AddPeriodicTimer(const Timestamp timestamp_scheduled, const Duration &duration_periodicity,
                          const Timestamp timestamp_end, F function_callback,
                          P pointer_callback_class, void *structure);

    // add order manager
    void AddOrderManager(OrderManager *manager);

    // add command listener
    void AddCommandRelay(CommandRelay *relay);

    // set stop timestamp
    void SetStopTimestamp(const Timestamp &timestamp);

    void SetMulticastReceiverManager(MulticastReceiverManager *multicast_manager)
    {
        multicast_manager_ = multicast_manager;
    }

    void SetRingBufferManager(RingBufferManager *ringbuffer_manager)
    {
        ringbuffer_manager_ = ringbuffer_manager;
    }

    // get current time
    const Timestamp &GetCurrentTime() const
    {
        return current_time_;
    }

    // get current spin counter
    size_t GetSpinCount() const
    {
        return spin_counter_;
    }

    const ObjectManager *GetObjectManager() const
    {
        return object_manager_;
    }

    // get global configuration
    const GlobalConfiguration *GetGlobalConfiguration() const
    {
        return configuration_;
    }

    // get date
    Date GetDate()
    {
        return date_;
    }

    // gets start and end of date
    Timestamp GetDayStart() const
    {
        return Timestamp::from_date_time(date_, "00:00:00.000000000");
    }

    Timestamp GetDayEnd() const
    {
        return Timestamp::from_date_time(date_, "23:59:59.999999999");
    }

    EngineEventLoopType GetEventLoopType() const
    {
        return event_loop_type_;
    }

    // run event loop
    void RunEventLoop();

    // stop event loop
    void EndEventLoop();

    // change date for engine across days
    void UpdateDate(const Date &date)
    {
        date_ = date;
    }

  private:
    const ObjectManager *      object_manager_;
    const GlobalConfiguration *configuration_;
    Date                       date_;

    // current time
    Timestamp current_time_;
    uint64_t  spin_counter_;

    // adds a timer source (timers should be handled encapsulatedly inside engine)
    void AddTimerSource(TimerSource *source);

    // Simulation event loop
    void SimulationRunEventLoop(const Timestamp &run_until_after_timestamp);

    void PeekAndCleanTimerSources(Timestamp &earliest_timestamp);

    // Production event loop
    void ProductionRunEventLoop();

    void PacketLogThread();

    // are we Simulation or Production
    EngineEventLoopType event_loop_type_;

    // are we running or not
    bool event_loop_running_;

#ifdef TIMER_SOURCES_WITH_MULTIMAP
    // timer_sources_timestamp
    std::multimap<Timestamp, TimerSource *> timer_sources_;
#else
    // timer sources to be handled
    std::vector<TimerSource *> timer_sources_;
#endif
    // order manager to be handled
    std::vector<OrderManager *> order_managers_;

    // command relays to be handled
    std::vector<CommandRelay *> command_relays_;

    MarketDataProviderManager market_data_provider_manager_;

    Timestamp stop_timestamp_;
    Timestamp first_timer_ts_;
    Timestamp dummy_timestamp_;

    MulticastReceiverManager *multicast_manager_;
    RingBufferManager *       ringbuffer_manager_;

    PacketLogStruct *packet_log_struct_;

    std::thread *packet_log_thread_;

    bool stop_packet_log_thread_;
};

template <typename F, typename P>
void Engine::AddPeriodicTimer(const Timestamp timestamp_scheduled,
                              const Duration &duration_periodicity, F function_callback,
                              P pointer_callback_class, void *data)
{
    TimerSource *timer_source{new TimerSource()};
    timer_source->AddTimerListener(function_callback, pointer_callback_class, data);

    const Date      date{Date::from_yyyymmdd(21201231)};
    const Timestamp timestamp_end{Timestamp::from_date(date)};
    timer_source->Schedule(timestamp_scheduled, duration_periodicity, timestamp_end);

    AddTimerSource(timer_source);
}

template <typename F, typename P>
void Engine::AddPeriodicTimer(const Timestamp timestamp_scheduled,
                              const Duration &duration_periodicity, const Timestamp endtime,
                              F function_callback, P pointer_callback_class, void *data)
{
    TimerSource *timer_source{new TimerSource()};
    timer_source->AddTimerListener(function_callback, pointer_callback_class, data);

    timer_source->Schedule(timestamp_scheduled, duration_periodicity, endtime);

    AddTimerSource(timer_source);
}

template <typename F, typename P>
void Engine::AddOneTimeTimer(const Timestamp timestamp_scheduled, F function_callback,
                             P pointer_callback_class, void *data)
{
    TimerSource *timer_source{new TimerSource()};
    timer_source->AddTimerListener(function_callback, pointer_callback_class, data);

    timer_source->Schedule(timestamp_scheduled);

    AddTimerSource(timer_source);
}

template <typename F, typename P>
void Engine::AddOneTimeTimer(const Timestamp timestamp_scheduled, F function_callback,
                             std::vector<P> pointer_callback_classes, void *data)
{
    auto timer_source = new TimerSource();
    for (auto &p : pointer_callback_classes)
        timer_source->AddTimerListener(function_callback, p, data);

    timer_source->Schedule(timestamp_scheduled);

    AddTimerSource(timer_source);
}

}  // namespace alphaone

#endif
