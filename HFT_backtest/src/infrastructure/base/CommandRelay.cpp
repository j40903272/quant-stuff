#include "infrastructure/base/CommandRelay.h"

namespace alphaone
{

void CommandRelay::Connect(const std::string &server)
{
    auto retry_count{0};
    auto status = connection_.Connect({server});
    while (!status && retry_count < maximum_retry_)
    {
        auto sleep_seconds = RandomInt();
        SPDLOG_WARN("[CommandRelay] Failed to connect to NATS server {}, going to retry #{} after "
                    "{} seconds",
                    retry_count++, server, sleep_seconds);
        sleep(sleep_seconds);
        status = connection_.Connect({server});
    }

    if (status)
    {
        SPDLOG_INFO("[CommandRelay] Successfully connect to {}", server);
    }
    else
    {
        SPDLOG_ERROR(
            "[CommandRelay] Failed to connect to {} after maximum retry {}, going to abort", server,
            maximum_retry_);
        abort();
    }
}

void CommandRelay::Subscribe(const std::string &subject)
{

    channel_ = subject;
    connection_.SubscribeAsync(channel_,
                               std::bind(&CommandRelay::OnMsg, this, std::placeholders::_1,
                                         std::placeholders::_2, std::placeholders::_3,
                                         std::placeholders::_4),
                               this);
}

// void CommandRelay::Response(const std::string &response)
// {
//     MessageFormat::MsgDataStruct *msgdata{new MessageFormat::MsgDataStruct{}};
//     (*msgdata->mutable_stringmap())["response"] = response;
//     connection_.PublishMsgData(channel_ + ".response", msgdata);
// }

// void CommandRelay::Emit(const std::string &response)
// {
//     MessageFormat::MsgDataStruct *msgdata{new MessageFormat::MsgDataStruct{}};
//     (*msgdata->mutable_stringmap())["response"] = response;
//     connection_.PublishMsgData(channel_ + ".emit", msgdata);
// }

void CommandRelay::AddCommandListener(CommandListener *listener)
{
    if (auto it = std::find(listeners_.begin(), listeners_.end(), listener) == listeners_.end())
    {
        // listener->SetCommandRelay(this);
        listeners_.push_back(listener);
    }
}

void CommandRelay::OnMsg(natsConnection *connection, natsSubscription *subscription,
                         natsMsg *message, void *data)
{
    std::lock_guard<std::mutex> lock(is_on_mutex_);
    // messages_[write_index_++].ParseFromArray(natsMsg_GetData(message),
    //                                          natsMsg_GetDataLength(message));
    ++read_index_;
    if (BRANCH_UNLIKELY(write_index_ >= BUFFER_COUNT))
    {
        SPDLOG_WARN("Message count exceeds buffer size = {}, start to discard oldest message",
                    BUFFER_COUNT);
        write_index_ = 0;
    }
}

void CommandRelay::Process(const Timestamp &event_loop_time)
{
    std::lock_guard<std::mutex> lock(is_on_mutex_);
    const auto                  read_index{std::min(BUFFER_COUNT, read_index_)};
    for (int i = 0; i < read_index; ++i)
    {
        // for (auto &l : listeners_)
        // {
        //     l->OnCommand(event_loop_time, messages_[i]);
        // }
    }
    // reset index after processing
    read_index_  = 0;
    write_index_ = 0;
}

}  // namespace alphaone
