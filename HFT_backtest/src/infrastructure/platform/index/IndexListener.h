#ifndef _INDEXLISTENER_H_
#define _INDEXLISTENER_H_

#include "infrastructure/common/datetime/Timestamp.h"

namespace alphaone
{

class IndexMessage;

class IndexListener
{
  public:
    virtual void OnUpdate(const Timestamp &event_loop_time, const IndexMessage *im) = 0;
};

}  // namespace alphaone


#endif