#ifndef _SEQUENCE_H_
#define _SEQUENCE_H_

#include "infrastructure/common/typedef/Typedefs.h"

#include <cstdint>

namespace alphaone
{

int64_t ParseSequenceNumber(DataSourceID data_source_id, void *raw_packet);

}  // namespace alphaone


#endif