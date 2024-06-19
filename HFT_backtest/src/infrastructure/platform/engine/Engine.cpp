#include "Engine.h"

#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/datetime/Timestamp.h"
// #include "infrastructure/common/protobuf/MessageFormat.pb.h"
#include "infrastructure/common/util/Logger.h"
#include "infrastructure/common/util/LoggerFormat.h"

#include <filesystem>
#include <iostream>

namespace alphaone
{
Engine::Engine(const ObjectManager *object_manager, const Date &date,
               EngineEventLoopType event_loop_type)
    : object_manager_{object_manager}
    , configuration_{object_manager->GetGlobalConfiguration()}
    , date_{date}
    , current_time_{Timestamp::invalid()}
    , spin_counter_{0}
    , event_loop_type_{event_loop_type}
    , event_loop_running_{false}
    , market_data_provider_manager_{event_loop_type, object_manager}
    , stop_timestamp_{Timestamp::max_time()}
    , first_timer_ts_{Timestamp::invalid()}
    , dummy_timestamp_{Timestamp::invalid()}
    , multicast_manager_{nullptr}
    , ringbuffer_manager_{nullptr}
    , packet_log_struct_{nullptr}
    , packet_log_thread_{nullptr}
    , stop_packet_log_thread_{true}
{
    if (GetGlobalConfiguration()->GetJson().contains("/System/packetlog"_json_pointer) &&
        GetGlobalConfiguration()->GetJson()["System"]["packetlog"].get<bool>() == true)
    {
        SPDLOG_INFO("[{}] Enable Packetlog Utility, goint to create extra thread", __func__);
        stop_packet_log_thread_ = false;
        packet_log_struct_      = new PacketLogStruct();
        packet_log_struct_->RingBuffer_PacketLog =
            new twone::RingBuffer(0, 0, sizeof(PacketLogStruct), 32768, -1, 0);
        packet_log_thread_ = new std::thread(std::bind(&Engine::PacketLogThread, this));
    }
    market_data_provider_manager_.SetPacketLogStruct(packet_log_struct_);
}

Engine::~Engine()
{
    for (auto &timer_source : timer_sources_)
    {
#ifdef TIMER_SOURCES_WITH_MULTIMAP
        delete timer_source.second;
        timer_source.second = nullptr;
#else
        delete timer_source;
        timer_source = nullptr;
#endif
    }
    timer_sources_.clear();

    stop_packet_log_thread_ = true;

    if (packet_log_struct_ != nullptr)
    {
        if (packet_log_struct_->RingBuffer_PacketLog != nullptr)
        {
            delete packet_log_struct_->RingBuffer_PacketLog;
            packet_log_struct_->RingBuffer_PacketLog = nullptr;
        }
        delete packet_log_struct_;
        packet_log_struct_ = nullptr;
    }

    if (packet_log_thread_ != nullptr)
    {
        packet_log_thread_->join();
        delete packet_log_thread_;
        packet_log_thread_ = nullptr;
    }
}

bool Engine::IsSimulation() const
{
    return (event_loop_type_ != EngineEventLoopType::Production);
}

void Engine::AddMarketDataListener(const Symbol *symbol, MarketDataListener *listener)
{
    std::cout << "AddMarketDataListener" << std::endl;
    market_data_provider_manager_.AddMarketDataListener(symbol, listener);
}

void Engine::AddTimerSource(TimerSource *source)
{
#ifdef TIMER_SOURCES_WITH_MULTIMAP
    timer_sources_.insert({source->PeekTimestamp(), source});
#else
    if (std::find(timer_sources_.begin(), timer_sources_.end(), source) == timer_sources_.end())
    {
        timer_sources_.push_back(source);
    }
#endif
}

void Engine::AddOneTimeTimer(const Timestamp timestamp_scheduled, TimerListener *listener,
                             void *data)
{
    TimerSource *timer_source{new TimerSource()};
    timer_source->AddTimerListener(listener, data);

    timer_source->Schedule(timestamp_scheduled);

    AddTimerSource(timer_source);
}


void Engine::AddPeriodicTimer(const Timestamp timestamp_scheduled,
                              const Duration &duration_periodicity, TimerListener *listener,
                              void *data)
{
    TimerSource *timer_source{new TimerSource()};
    timer_source->AddTimerListener(listener, data);

    const Date      date{Date::from_yyyymmdd(21201231)};
    const Timestamp timestamp_end{Timestamp::from_date(date)};
    timer_source->Schedule(timestamp_scheduled, duration_periodicity, timestamp_end);

    AddTimerSource(timer_source);
}

void Engine::AddPeriodicTimer(const Timestamp timestamp_start, const Duration &duration_periodicity,
                              const Timestamp timestamp_end, TimerListener *listener, void *data)
{
    TimerSource *timer_source{new TimerSource()};
    timer_source->AddTimerListener(listener, data);

    timer_source->Schedule(timestamp_start, duration_periodicity, timestamp_end);

    AddTimerSource(timer_source);
}

void Engine::AddOrderManager(OrderManager *manager)
{
    if (manager != nullptr)
    {
        manager->SetPacketLogStruct(packet_log_struct_);
    }

    if (std::find(order_managers_.begin(), order_managers_.end(), manager) == order_managers_.end())
    {
        order_managers_.push_back(manager);
    }
}

void Engine::AddCommandRelay(CommandRelay *relay)
{
    if (std::find(command_relays_.begin(), command_relays_.end(), relay) == command_relays_.end())
    {
        command_relays_.push_back(relay);
    }
}

void Engine::SetStopTimestamp(const Timestamp &timestamp)
{
    stop_timestamp_ = timestamp;
}


void Engine::RunEventLoop()
{
    market_data_provider_manager_.SortMarketDataListener();
    event_loop_running_ = true;

    if (event_loop_type_ == EngineEventLoopType::Simulation)
    {
        SimulationRunEventLoop(Timestamp::max_time());
    }
    else if (event_loop_type_ == EngineEventLoopType::Production)
    {
        ProductionRunEventLoop();
    }
    else
    {
        SPDLOG_ERROR("invalid event loop type");
        abort();
    }
}

void Engine::EndEventLoop()
{
    event_loop_running_ = false;
}

void Engine::SimulationRunEventLoop(const Timestamp &run_until_after_timestamp)
{
    while (event_loop_running_)
    {
        // std::cout << "SimulationRunEventLoop" << std::endl;
        Timestamp earliest_timestamp{Timestamp::max_time()};

        // check market data sources for earliest timestamp
        earliest_timestamp = market_data_provider_manager_.PeekTimestamp();
        std::cout << "earliest_timestamp: " << earliest_timestamp << std::endl;
        PeekAndCleanTimerSources(earliest_timestamp);
        SPDLOG_INFO("earliest_timestamp: {}, stop_timestamp_ {}", earliest_timestamp, stop_timestamp_);
        if (earliest_timestamp > stop_timestamp_)
        {
            return;
        }

        std::cout << "run_until_after_timestamp: " << run_until_after_timestamp << std::endl;
        if (earliest_timestamp >= run_until_after_timestamp)
        {
            if (run_until_after_timestamp == Timestamp::max_time() ||
                earliest_timestamp == Timestamp::max_time())
            {
                return;
            }
            else
            {
                stop_timestamp_ = earliest_timestamp;
            }
        }

        // process
        ++spin_counter_;
        current_time_ = earliest_timestamp;
        std::cout << "Processssssssss" << std::endl;
        market_data_provider_manager_.Process(earliest_timestamp);

        // get data_sources_ size
        // SPDLOG_INFO("fkfkfkfkfkfkkffk data_sources_ size: {}", data_sources_.size());

#ifdef TIMER_SOURCES_WITH_MULTIMAP
        first_timer_ts_ = timer_sources_.empty() ? first_timer_ts_ : timer_sources_.begin()->first;
        for (auto &timer_source : timer_sources_)
        {
            if (timer_source.first != first_timer_ts_)
                break;

            timer_source.second->Process(earliest_timestamp);
        }
#else
        for (auto &timer_source : timer_sources_)
        {
            timer_source->Process(earliest_timestamp);
        }
#endif
        for (auto &order_manager : order_managers_)
        {
            order_manager->Process(earliest_timestamp);
        }
    }
}

void Engine::PeekAndCleanTimerSources(Timestamp &earliest_timestamp)
{
    if (timer_sources_.empty())
        return;

        // check timer sources for earliest timestamp and clean timer sources
#ifdef TIMER_SOURCES_WITH_MULTIMAP
    auto it{timer_sources_.begin()};
    auto first_timer_source_ts{it->first};
    for (; it != timer_sources_.end() && it->first == first_timer_source_ts;)
    {
        Timestamp peeked_timestamp{it->second->PeekTimestamp()};
        if (!peeked_timestamp.is_valid())
        {
            delete it->second;
            it = timer_sources_.erase(it);
            if (it != timer_sources_.end())
            {
                first_timer_source_ts = it->first;
            }
        }
        else
        {
            if (it->first != peeked_timestamp)
            {
                timer_sources_.insert({peeked_timestamp, it->second});
                it = timer_sources_.erase(it);
            }
            else
            {
                ++it;
            }
            if (peeked_timestamp < earliest_timestamp)
            {
                earliest_timestamp = peeked_timestamp;
            }
        }
    }
#else
    auto it{timer_sources_.begin()};
    while (it != timer_sources_.end())
    {
        Timestamp peeked_timestamp{(*it)->PeekTimestamp()};
        if (!peeked_timestamp.is_valid())
        {
            delete *it;
            it = timer_sources_.erase(it);
        }
        else
        {
            if (peeked_timestamp < earliest_timestamp)
            {
                earliest_timestamp = peeked_timestamp;
            }
            ++it;
        }
    }
#endif
}

void Engine::ProductionRunEventLoop()
{
    while (event_loop_running_)
    {
        // process
        ++spin_counter_;
        current_time_ = Timestamp::now();

        market_data_provider_manager_.Process(current_time_);

        for (auto &order_manager : order_managers_)
        {
            order_manager->Process(current_time_);
        }

        for (auto &command_relay : command_relays_)
        {
            command_relay->Process(current_time_);
        }

        if (multicast_manager_)
        {
            multicast_manager_->Process(current_time_);
        }

        if (ringbuffer_manager_)
        {
            ringbuffer_manager_->Process(current_time_);
        }

        PeekAndCleanTimerSources(dummy_timestamp_);
#ifdef TIMER_SOURCES_WITH_MULTIMAP
        first_timer_ts_ = timer_sources_.empty() ? first_timer_ts_ : timer_sources_.begin()->first;
        for (auto &timer_source : timer_sources_)
        {
            if (timer_source.first != first_timer_ts_)
                break;

            timer_source.second->Process(current_time_);
        }
#else
        for (auto &timer_source : timer_sources_)
        {
            timer_source->Process(current_time_);
        }
#endif
    }
}

void Engine::PacketLogThread()
{
    const auto &j = GetGlobalConfiguration()->GetJson();
    if (!j.contains("/System/nats_server"_json_pointer))
    {
        SPDLOG_ERROR("No available nats server, stop {}", __func__);
        return;
    }
    NATSConnection nats_connection;
    nats_connection.Connect(
        std::vector<std::string>{j["System"]["nats_server"].get<std::string>()});

    while (!stop_packet_log_thread_)
    {
        void *data = NULL;
        while (packet_log_struct_->RingBuffer_PacketLog->SequentialGet(&data))
        {
            PacketLogStruct *packet_log_struct = (PacketLogStruct *)data;

            if (packet_log_struct->Type == (int)PacketLogType::TAIFEX ||
                packet_log_struct->Type == (int)PacketLogType::TWSE)
            {
                continue;
                // MessageFormat::MsgDataStruct *msgdata = new MessageFormat::MsgDataStruct();

                // char buf[1024];
                // sprintf(buf, "%d", packet_log_struct->ChannelID);
                // (*msgdata->mutable_stringmap())[std::string("ChannelID")] = buf;

                // if (packet_log_struct->ChannelID == (int)PacketLogChannelID::FUTURE ||
                //     packet_log_struct->ChannelID == (int)PacketLogChannelID::OPTION)
                // {
                //     sprintf(buf, "%d", Decode5(packet_log_struct->ChannelSeq));
                //     (*msgdata->mutable_stringmap())[std::string("SeqNumber")] = buf;
                // }
                // else if (packet_log_struct->ChannelID == (int)PacketLogChannelID::TSE ||
                //          packet_log_struct->ChannelID == (int)PacketLogChannelID::OTC)
                // {
                //     sprintf(buf, "%d", Decode4(packet_log_struct->ChannelSeq));
                //     (*msgdata->mutable_stringmap())[std::string("SeqNumber")] = buf;
                // }


                // sprintf(buf, "%d", packet_log_struct->SeqNum);
                // (*msgdata->mutable_stringmap())[std::string("OrderIndex")] = buf;

                // sprintf(buf, "%d", packet_log_struct->LoopCount);
                // (*msgdata->mutable_stringmap())[std::string("LoopCount")] = buf;

                // if (packet_log_struct->ExecType == '0')
                // {
                //     (*msgdata->mutable_stringmap())[std::string("OrderNo")] =
                //         std::string(packet_log_struct->OrderNo, 5);
                // }
                // else
                // {
                //     (*msgdata->mutable_stringmap())[std::string("OrderNo")] =
                //         std::string(packet_log_struct->OrderNoStr, 5);
                // }

                // const char *user_define = packet_log_struct->UserDefine;

                // (*msgdata->mutable_stringmap())[std::string("Tid")] =
                //     std::string(&user_define[0], 4);

                // (*msgdata->mutable_stringmap())[std::string("SysID")] =
                //     std::string(&user_define[4], 2);

                // sprintf(buf, "%d", packet_log_struct->OrderTo);
                // (*msgdata->mutable_stringmap())[std::string("OrderTo")] = buf;

                // (*msgdata->mutable_stringmap())[std::string("ExecType")] =
                //     packet_log_struct->ExecType;

                // nats_connection.PublishMsgData("protocol.ordermapping", msgdata);
            }
        }
        sleep(1);
    }
}

}  // namespace alphaone
