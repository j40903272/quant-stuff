#ifndef _DELAY_H_
#define _DELAY_H_

#include <cstdint>

namespace alphaone
{

struct DelayInfo
{
    int64_t seqno_;
    char    orderno_[5];
    int64_t feedts_;
    int64_t r01ts_;
    int64_t r02ts_;
};

}  // namespace alphaone

#endif