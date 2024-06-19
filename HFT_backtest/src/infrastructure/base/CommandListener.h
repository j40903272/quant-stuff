#ifndef _COMMANDLISTENER_H_
#define _COMMANDLISTENER_H_

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/nats/NATSConnection.h"
#include "infrastructure/common/util/Logger.h"

#include <iostream>

#define BUFFER_COUNT 64

namespace alphaone
{

class CommandRelay;

class CommandListener
{
  public:
    CommandListener()                        = default;
    CommandListener(const CommandListener &) = delete;
    CommandListener &operator=(const CommandListener &) = delete;

    virtual ~CommandListener() = default;

    // command callback
    // virtual void OnCommand(const Timestamp &                    event_loop_time,
    //                        const MessageFormat::GenericMessage &gm) = 0;
    // virtual void SetCommandRelay(CommandRelay *relay)               = 0;
};
}  // namespace alphaone

#endif
