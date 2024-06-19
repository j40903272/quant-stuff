#ifndef _MARKET_DATA_PROVIDER_TWSE_H_
#define _MARKET_DATA_PROVIDER_TWSE_H_

#include "infrastructure/base/MarketDataListener.h"
#include "infrastructure/base/MarketDataProvider.h"
#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/message/TWSE.h"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/twone/def/Def.h"
#include "infrastructure/common/twone/ringbuffer/RingBuffer.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/PacketLog.h"
#include "infrastructure/common/util/UnPackBCD.h"
#include "infrastructure/platform/book/MarketByOrderBook.h"
#include "infrastructure/platform/datasource/MarketDataSource.h"

#include <string.h>
#include <tuple>
#include <vector>

namespace alphaone
{

class MarketDataProvider_TWSE : public MarketDataProvider
{
  public:
    MarketDataProvider_TWSE(DataSourceID data_source_id);
    ~MarketDataProvider_TWSE();
    void Init();

    void            Process(const Timestamp &event_loop_time);
    void            ProcessPacket(void *data);
    const Timestamp PeekTimestamp();

    DataSourceType GetType() const
    {
        return DataSourceType::MarketByPrice;
    }

    ProviderID GetProviderID() const;

    void             SetPacketLogStruct(PacketLogStruct *packet_log_struct);
    PacketLogStruct *GetPacketLogStruct();

  private:
    MarketDataMessage marketdata_message_;
    MarketDataSource *market_data_source_;
    unsigned int      timestamp_offset_;

    twone::RingBuffer rb_marketdata_;
    twone::RingBuffer rb_warrant_;

    DataSourceID data_source_id_;

    PacketLogStruct *packet_log_struct_;
    int              packet_log_channel_id_;

    std::vector<std::tuple<const Symbol *, void *>> twse_orderbook_list_;

    void Notify(std::vector<MarketDataListener *> &list, MarketDataMessage *msg, void *raw_packet);

    void ParseFormat6(TWSEDataFormat6_RealTime_t *packet, Timestamp &ts);
};
}  // namespace alphaone
#endif
