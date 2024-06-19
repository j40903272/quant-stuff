#include "CommandSource.h"

namespace alphaone
{
CommandSource::CommandSource(const std::string &server, const std::string &object,
                             const double delay)
    : server_{server}
    , object_{object}
    , delay_{delay}
    , connection_{std::make_shared<NATSConnection>()}
    , maximum_retry_{10}
{
    auto retry_count{0};
    auto status = connection_->Connect({server_});
    while (!status && retry_count < maximum_retry_)
    {
        SPDLOG_WARN("[CommandSource] Failed to connect to NATS server {}, going to retry #{} after "
                    "{} seconds",
                    retry_count++, server_, 1);
        sleep(1);
        status = connection_->Connect({server_});
    }

    if (status)
        SPDLOG_INFO("[CommandSource] Successfully connect to {}, with object = {}", server_,
                    object_);
    else
        SPDLOG_ERROR("[CommandSource] Failed to connect sql connection to {}", server_);
}

CommandSource::CommandSource(std::shared_ptr<NATSConnection> conn, const std::string &object,
                             const double delay)
    : object_{object}, delay_{delay}, maximum_retry_{10}
{
    if (!conn)
        throw std::invalid_argument("conn is not initialized");

    connection_ = conn;
}

CommandSource::~CommandSource()
{
}

// void CommandSource::Publish(const std::string &word)
// {
//     MessageFormat::MsgDataStruct *msgdata{new MessageFormat::MsgDataStruct{}};
//     (*msgdata->mutable_stringmap())["word"] = word;
//     connection_->PublishMsgData(object_, msgdata);
// }

// void CommandSource::Publish(const nlohmann::json &js)
// {
//     if (js.is_array())
//     {
//         for (const auto &j : js)
//         {
//             PublishSingleJson(j);
//             sleep(delay_);
//         }
//     }
//     else
//     {
//         PublishSingleJson(js);
//     }
// }

// void CommandSource::PublishSingleJson(const nlohmann::json &js)
// {
//     MessageFormat::MsgDataStruct *msgdata{new MessageFormat::MsgDataStruct{}};
//     for (const auto &[ik, iv] : js.items())
//     {
//         std::stringstream key;
//         key << ik;

//         std::stringstream value;
//         if (iv.is_string())
//             value << iv.get<std::string>();
//         else
//             value << iv;

//         (*msgdata->mutable_stringmap())[key.str()] = value.str();
//     }
//     connection_->PublishMsgData(object_, msgdata);
// }
}  // namespace alphaone
