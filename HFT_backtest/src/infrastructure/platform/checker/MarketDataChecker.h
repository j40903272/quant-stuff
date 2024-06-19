#ifndef _MARKETDATACHECKER_H
#define _MARKETDATACHECKER_H

#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/common/datetime/Duration.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/util/EnumToString.h"
#include "infrastructure/platform/reader/MarketDataReader.h"

#include <boost/circular_buffer.hpp>
#include <unordered_map>
#include <vector>

namespace alphaone
{

enum CheckingMessageType
{
    CheckBook               = 0,
    CheckTrade              = 1,
    CheckPacketEnd          = 2,
    LastCheckingMessageType = 3,
};

struct MarketDataMessageCache
{
    MarketDataMessageCache(const MarketDataMessage *m)
    {
        symbol_          = m->symbol;
        mtype_           = m->market_data_message_type;
        provider_time_   = m->provider_time;
        sequence_number_ = m->sequence_number;
        mbp_             = m->mbp;
        mbo_             = m->mbo;
        trade_           = m->trade;
        type_            = m->type;
    }
    const Symbol *          symbol_;
    MarketDataMessageType   mtype_;
    Timestamp               provider_time_;
    int64_t                 sequence_number_;
    MarketDataMessage_MBP   mbp_;
    MarketDataMessage_MBO   mbo_;
    MarketDataMessage_Trade trade_;
    DataSourceType          type_;

    friend bool operator==(const MarketDataMessageCache &c1, const MarketDataMessageCache &c2);
    friend bool operator!=(const MarketDataMessageCache &c1, const MarketDataMessageCache &c2);
};

class MarketDataChecker : public MarketDataReader
{
  public:
    MarketDataChecker(const ObjectManager *object_manager, const bool is_light_mode);
    ~MarketDataChecker();
    void        PrepareTimestampSeq();
    void        OnMarketDataMessage(const MarketDataMessage *mdm, void *raw_packet);
    void        CheckMbpValidity(const MarketDataMessage *mdm);
    void        CheckMboValidity(const MarketDataMessage *mdm);
    void        CheckTradeValidity(const MarketDataMessage *mdm);
    void        CheckDuplicateMsg(const MarketDataMessage *mdm, const int provider_id,
                                  const int symbol_id);
    int         GetCheckingSymbolId(const Symbol *);
    std::string GetDataSourceIDString(const DataSourceID &data_source_id);
    std::string GetMessageTypeString(const MarketDataMessageType &type);

    std::string CheckingTypeToString(const CheckingMessageType &c_type)
    {
        switch (c_type)
        {
        case CheckingMessageType::CheckBook:
            return "Book";
        case CheckingMessageType::CheckTrade:
            return "Trade";
        case CheckingMessageType::CheckPacketEnd:
            return "PacketEnd";
        default:
            return "UNDEFINED";
        }
    }


  private:
    nlohmann::json                          json_;
    std::vector<Duration>                   intervals_[BookTradePart];
    const int64_t                           squence_number_max_diff_;
    const int                               check_layer_;
    const int                               message_buffer_size_;
    const bool                              is_check_trade_distribution_;
    const bool                              is_light_mode_;
    BookPrice                               touch_price_[AskBid];
    BookQty                                 touch_qty_[AskBid];
    std::vector<const Symbol *>             checking_symbols_;
    std::unordered_map<const Symbol *, int> checking_symbol_to_symbol_id_map_;
    std::vector<std::vector<Timestamp>>     last_event_time_[BookTradePart];
    std::vector<std::vector<size_t>>        message_count_[LastCheckingMessageType];
    std::vector<std::vector<int64_t>>       last_seq_;

    std::vector<std::vector<std::map<BookPrice, int>>>                       price_count_;
    std::vector<std::vector<boost::circular_buffer<MarketDataMessageCache>>> checking_msgs_;

    Timestamp last_valid_message_ts_[static_cast<size_t>(DataSourceID::END)];
};


}  // namespace alphaone
#endif
