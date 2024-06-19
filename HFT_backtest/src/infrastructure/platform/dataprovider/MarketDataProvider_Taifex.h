#ifndef _MARKET_DATA_PROVIDER_TAIFEX_H_
#define _MARKET_DATA_PROVIDER_TAIFEX_H_

#include "infrastructure/base/MarketDataListener.h"
#include "infrastructure/base/MarketDataProvider.h"
#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/message/TAIFEX.h"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/twone/def/Def.h"
#include "infrastructure/common/twone/orderbook/TaifexOrderBook.h"
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

class MarketDataProvider_Taifex : public MarketDataProvider

{
  public:
    MarketDataProvider_Taifex(DataSourceID data_source_id);
    ~MarketDataProvider_Taifex();
    void Init();

    void            Process(const Timestamp &event_loop_time);
    const Timestamp PeekTimestamp();

    DataSourceType GetType() const
    {
        return DataSourceType::MarketByPrice;
    }

    ProviderID GetProviderID() const;

    void AddMarketDataListener(const Symbol *symbol, MarketDataListener *listener);


    void             SetPacketLogStruct(PacketLogStruct *packet_log_struct);
    PacketLogStruct *GetPacketLogStruct();

  private:
    char         transmission_code;
    bool         snapshot_finish_;
    int          begin_sequence_number_;
    unsigned int timestamp_offset_;

    MarketDataSource *market_data_source_;

    MarketDataMessage marketdata_message_;

    DataSourceID data_source_id_;

    twone::RingBuffer rb_marketdata_;
    twone::RingBuffer rb_marketdata_snapshot_;
    twone::RingBuffer rb_marketdata_rewind_;

    std::vector<std::tuple<const Symbol *, twone::TaifexOrderBook *>> taifex_orderbook_list_;

    PacketLogStruct *packet_log_struct_;

    void ParseI080(TXMarketDataI080_t *pI080);
    void ParseI020(TXMarketDataI020_t *pI020);
    bool ProcessSnapshot();

    void Notify(std::vector<MarketDataListener *> &list, MarketDataMessage *msg, void *raw_packet);

    void ParseI081(TXMarketDataI081_t *pI081, bool notify, bool isrewind, Timestamp &ts);
    void ParseI083(TXMarketDataI083_t *pI083, bool notify, bool isrewind, Timestamp &ts);
    void ParseI024(TXMarketDataI024_t *pI024, bool notify, bool isrewind, Timestamp &ts);
    void ParseI025(TXMarketDataI025_t *pI025, bool notify, bool isrewind, Timestamp &ts);
};
}  // namespace alphaone
#endif
