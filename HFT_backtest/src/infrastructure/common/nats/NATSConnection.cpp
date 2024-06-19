#include "NATSConnection.h"

#include <sstream>

namespace alphaone
{

void OnMsg(natsConnection *nc, natsSubscription *sub, natsMsg *msg, void *data);

NATSConnection::NATSConnection() : connection_{nullptr}, options_{nullptr}, is_connected_{false}
{
}

NATSConnection::~NATSConnection()
{
    for (auto it = subscriptions_.begin(); it != subscriptions_.end(); ++it)
    {
        delete it->second;
        // natsSubscription_Destroy(it->first);
    }

    if (connection_)
    {
        // natsConnection_Close(connection_);
        // natsConnection_Destroy(connection_);
    }
    // natsOptions_Destroy(options_);
}

bool NATSConnection::Connect(std::vector<std::string> server_list)
{
    // natsStatus status;
    // status = natsOptions_Create(&options_);
    // if (status != NATS_OK)
    // {
    //     return false;
    // }

    std::vector<const char *> char_array(server_list.size());
    std::transform(server_list.begin(), server_list.end(), char_array.begin(),
                   std::mem_fun_ref(&std::string::c_str));

    // status = natsOptions_SetServers(options_, (const char **)char_array.data(), char_array.size());

    // if (status != NATS_OK)
    // {
    //     SPDLOG_WARN("[natsOptions_SetServers] {} natsStatus {} is not NATS_OK",
    //                 fmt::join(server_list, ","), status);
    //     return false;
    // }

    // natsOptions_SetAllowReconnect(options_, true);
    // natsOptions_SetMaxReconnect(options_, 999999);
    // natsOptions_SetReconnectWait(options_, 200);

    // status = natsConnection_Connect(&connection_, options_);

    // if (status != NATS_OK)
    // {
    //     SPDLOG_WARN("[natsConnection_Connect] {} natsStatus {} is not NATS_OK",
    //                 fmt::join(server_list, ","), status);
    //     return false;
    // }

    is_connected_      = true;
    connected_servers_ = std::unordered_set<std::string>(server_list.begin(), server_list.end());

    return true;
}

bool NATSConnection::IsConnectTo(const std::string &server)
{
    return connected_servers_.find(server) != connected_servers_.end();
}

// void NATSConnection::PublishMsgData(std::string subject, MessageFormat::MsgDataStruct *msgdata)
void NATSConnection::PublishMsgData(std::string subject, nlohmann::json *msgdata)
{
    if (is_connected_ && connection_)
    {
        // MessageFormat::GenericMessage gm;
        // gm.set_allocated_msgdata(msgdata);
        // int   length = gm.ByteSize();
        // char *buf    = new char[length];
        // gm.SerializeToArray(buf, length);
        // natsConnection_Publish(connection_, subject.c_str(), buf, length);
        // natsConnection_Flush(connection_);
        // delete[] buf;
    }
}

void NATSConnection::SubscribeAsync(
    std::string                                                                  subject,
    std::function<void(natsConnection *, natsSubscription *, natsMsg *, void *)> callback,
    void *                                                                       data)
{
    if (is_connected_ && connection_)
    {
        natsSubscription *subscription = nullptr;

        NATSCallbackObject *obj = new NATSCallbackObject();
        obj->callback           = callback;
        obj->data               = data;

        // natsStatus status =
        //     natsConnection_Subscribe(&subscription, connection_, subject.c_str(), OnMsg, obj);

        // if (status != NATS_OK)
        // {
        //     delete obj;
        //     // SPDLOG_INFO("[{}] status={}\n", __func__, natsStatus_GetText(status));
        // }
        // else
        // {
        //     subscriptions_[subscription] = obj;
        //     // natsConnection_Flush(connection_);
        // }
    }
}

// MessageFormat::GenericMessage *
// NATSConnection::Request(std::string subject, MessageFormat::GenericMessage *gm, int timeoutms)
// {
//     if (is_connected_ && connection_)
//     {
//         natsMsg *replyMsg = nullptr;

//         int   length = gm->ByteSize();
//         char *buf    = new char[length];
//         gm->SerializeToArray(buf, length);

//         natsStatus s =
//             natsConnection_Request(&replyMsg, connection_, subject.c_str(), buf, length, timeoutms);
//         if (s != NATS_OK)
//         {
//             delete[] buf;
//             natsMsg_Destroy(replyMsg);
//             return nullptr;
//         }

//         int         len  = natsMsg_GetDataLength(replyMsg);
//         const char *data = natsMsg_GetData(replyMsg);

//         std::stringstream stream;
//         stream.rdbuf()->pubsetbuf((char *)data, len);
//         MessageFormat::GenericMessage *ret = new MessageFormat::GenericMessage();
//         ret->ParseFromIstream(&stream);

//         delete[] buf;
//         natsMsg_Destroy(replyMsg);

//         return ret;
//     }
//     return nullptr;
// }

void OnMsg(natsConnection *nc, natsSubscription *sub, natsMsg *msg, void *user_data)
{
    if (user_data != nullptr)
    {
        NATSCallbackObject *obj = (NATSCallbackObject *)user_data;
        obj->callback(nc, sub, msg, obj->data);
    }
    // natsMsg_Destroy(msg);
}

}  // namespace alphaone
