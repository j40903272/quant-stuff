#ifndef _NATSCONNECTION_H_
#define _NATSCONNECTION_H_

// #include "infrastructure/common/protobuf/MessageFormat.pb.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/spdlog/spdlog.h"

#include <algorithm>
#include <functional>
#include <nats/nats.h>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace alphaone
{

struct NATSCallbackObject
{
    std::function<void(natsConnection *, natsSubscription *, natsMsg *, void *)> callback;
    void *                                                                       data;
};

class NATSConnection
{
  public:
    NATSConnection();
    ~NATSConnection();

    bool Connect(std::vector<std::string> server_list);
    bool IsConnectTo(const std::string &server);
    void SubscribeAsync(
        std::string                                                                  subject,
        std::function<void(natsConnection *, natsSubscription *, natsMsg *, void *)> callback,
        void *                                                                       user_data);

    // void PublishMsgData(std::string subject, MessageFormat::MsgDataStruct *msgdata);
    void PublishMsgData(std::string subject, nlohmann::json *msgdata);
    // MessageFormat::GenericMessage *Request(std::string subject, MessageFormat::GenericMessage *gm,
    //                                        int timeoutms);

  private:
    natsConnection *                                             connection_;
    natsOptions *                                                options_;
    std::unordered_map<natsSubscription *, NATSCallbackObject *> subscriptions_;
    bool                                                         is_connected_;
    std::unordered_set<std::string>                              connected_servers_;
};
}  // namespace alphaone
#endif
