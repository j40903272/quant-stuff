#include "Sequence.h"

#include "infrastructure/common/message/TAIFEX.h"
#include "infrastructure/common/twone/def/Def.h"
#include "infrastructure/common/util/UnPackBCD.h"

#include <cstdint>

namespace alphaone
{

int64_t ParseSequenceNumber(DataSourceID data_source_id, void *raw_packet)
{
    if (data_source_id == DataSourceID::TAIFEX_FUTURE ||
        data_source_id == DataSourceID::TAIFEX_OPTION)
    {
        TXMarketDataHdr_RealTime_t *packet = (TXMarketDataHdr_RealTime_t *)raw_packet;
        return Decode5(packet->ChannelSeq);
    }
    return -1;
}


}  // namespace alphaone
