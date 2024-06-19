#ifndef _COMMANDRELAY_H_
#define _COMMANDRELAY_H_

#include "infrastructure/base/CommandListener.h"
#include "infrastructure/base/CommandRelay.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/nats/NATSConnection.h"
#include "infrastructure/common/util/Logger.h"

#include <boost/algorithm/string.hpp>
#include <random>

#define BUFFER_COUNT 64

namespace alphaone
{
class CommandRelay
{
  public:
    CommandRelay()
        : channel_{""}, maximum_retry_{10}, write_index_{0}, read_index_{0}, distribution_(1, 5)
    {
        // messages_.resize(BUFFER_COUNT);
    }

    ~CommandRelay() = default;

    void Connect(const std::string &server);
    void Subscribe(const std::string &subject);
    void Response(const std::string &response);
    void Emit(const std::string &response);

    void AddCommandListener(CommandListener *listener);

    void OnMsg(natsConnection *connection, natsSubscription *subscription, natsMsg *message,
               void *data);

    void Process(const Timestamp &event_loop_time);

  private:
    std::string    channel_;
    NATSConnection connection_;
    int            maximum_retry_;
    int            write_index_;
    int            read_index_;
    std::mutex     is_on_mutex_;
    // message buffer
    // std::vector<MessageFormat::GenericMessage> messages_;
    // command listeners
    std::vector<CommandListener *> listeners_;

    // rng
    std::default_random_engine         generator_;
    std::uniform_int_distribution<int> distribution_;
    int                                RandomInt()
    {
        return distribution_(generator_);
    }
};

}  // namespace alphaone


#endif
