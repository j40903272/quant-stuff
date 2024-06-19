#include "MarketDataProvider_TWSE.h"

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/message/TWSE.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/Branch.h"
#include "infrastructure/common/util/Helper.h"
#include "infrastructure/platform/book/MarketByOrderBook.h"

#include <algorithm>

namespace alphaone
{
MarketDataProvider_TWSE::MarketDataProvider_TWSE(DataSourceID data_source_id)
    : MarketDataProvider{DataSourceType::MarketByPrice}
    , marketdata_message_{DataSourceType::MarketByPrice}
    , data_source_id_{data_source_id}
    , packet_log_struct_{nullptr}
    , packet_log_channel_id_{0}
{
    marketdata_message_.trade.is_not_duplicate_ = true;
    timestamp_offset_                           = sizeof(timespec);
}

MarketDataProvider_TWSE::~MarketDataProvider_TWSE()
{
}

void MarketDataProvider_TWSE::Init()
{
    if (data_source_id_ == DataSourceID::TSE)
    {
        rb_marketdata_ =
            twone::RingBuffer((int)twone::MARKETDATA_INDEX_RINGBUFFER_BASEINDEX::TSE_REATIME,
                              (int)twone::MARKETDATA_BLOCK_RINGBUFFER_BASEINDEX::TSE_REATIME,
                              MARKETDATA_TWSE_REALTIME_PACKET_SIZE,
                              MARKETDATA_TWSE_REALTIME_RINGBUFFER_LENGTH, -1, 0);
        rb_warrant_ = twone::RingBuffer(
            (int)twone::MARKETDATA_INDEX_RINGBUFFER_BASEINDEX::TSE_WARRANT_REATIME,
            (int)twone::MARKETDATA_BLOCK_RINGBUFFER_BASEINDEX::TSE_WARRANT_REATIME,
            MARKETDATA_TWSE_REALTIME_PACKET_SIZE, MARKETDATA_TWSE_REALTIME_RINGBUFFER_LENGTH, -1,
            0);
    }
    else if (data_source_id_ == DataSourceID::OTC)
    {
        rb_marketdata_ =
            twone::RingBuffer((int)twone::MARKETDATA_INDEX_RINGBUFFER_BASEINDEX::OTC_REATIME,
                              (int)twone::MARKETDATA_BLOCK_RINGBUFFER_BASEINDEX::OTC_REATIME,
                              MARKETDATA_TWSE_REALTIME_PACKET_SIZE,
                              MARKETDATA_TWSE_REALTIME_RINGBUFFER_LENGTH, -1, 0);
        rb_warrant_ = twone::RingBuffer(
            (int)twone::MARKETDATA_INDEX_RINGBUFFER_BASEINDEX::OTC_WARRANT_REATIME,
            (int)twone::MARKETDATA_BLOCK_RINGBUFFER_BASEINDEX::OTC_WARRANT_REATIME,
            MARKETDATA_TWSE_REALTIME_PACKET_SIZE, MARKETDATA_TWSE_REALTIME_RINGBUFFER_LENGTH, -1,
            0);
    }

    market_data_source_ = GetMarketDataSource(data_source_id_);
}

void MarketDataProvider_TWSE::Process(const Timestamp &event_loop_time)
{
    if (packet_log_struct_ != nullptr)
    {
        packet_log_struct_->ChannelID = packet_log_channel_id_;
    }

    void *data = NULL;
    while (rb_marketdata_.SequentialGet(&data))
        ProcessPacket(data);


    data = NULL;
    while (rb_warrant_.SequentialGet(&data))
        ProcessPacket(data);
}

void MarketDataProvider_TWSE::ProcessPacket(void *data)
{
    int       packetSize   = *(int *)data;
    int       offset       = 0;
    char *    pDataAddress = ((char *)data) + sizeof(int);
    timespec *timestamp =
        (timespec *)(((char *)data) + MARKETDATA_TWSE_REALTIME_PACKET_SIZE - timestamp_offset_);

    Timestamp ts =
        Timestamp::from_epoch_nsec((int64_t)(timestamp->tv_sec * 1000000000 + timestamp->tv_nsec));

    while (offset < packetSize)
    {
        char *         packet = pDataAddress + offset;
        TWSEDataHdr_t *hdr    = (TWSEDataHdr_t *)(packet);
        if (hdr->Format[0] == 0x06 || hdr->Format[0] == 0x17)
        {
            if (packet_log_struct_ != nullptr)
            {
                packet_log_struct_->SeqNum += 100;
            }
            ParseFormat6((TWSEDataFormat6_RealTime_t *)packet, ts);
        }
        offset += Decode2(hdr->BodyLength);
    }
}

void MarketDataProvider_TWSE::ParseFormat6(TWSEDataFormat6_RealTime_t *packet, Timestamp &ts)
{
    auto t = market_data_source_->GetMarketDataListenerNode(packet->ProductID, 6);

    if (BRANCH_LIKELY(t != nullptr))
    {
        if (!Format6IsSimulated(packet))
        {
            bool hasOrderBook = true;
            if ((*packet->InformationPrompt) & 1)
            {
                hasOrderBook = false;
            }

            int index = 0;
            if ((*packet->InformationPrompt) & TWSE_SHOW_MATCHPRICETMASK)
            {

                marketdata_message_.market_data_message_type =
                    MarketDataMessageType::MarketDataMessageType_Trade;
                marketdata_message_.symbol                      = std::get<0>(*t);
                marketdata_message_.provider_time               = ts;
                marketdata_message_.exchange_time               = Timestamp::from_epoch_nsec(0);
                marketdata_message_.sequence_number             = 0;
                marketdata_message_.trade.counterparty_order_id = 0;
                marketdata_message_.trade.order_id              = 0;
                marketdata_message_.trade.price         = Decode5(packet->Records[index].Price);
                marketdata_message_.trade.qty           = Decode4(packet->Records[index].Qty);
                marketdata_message_.trade.side          = 0;
                marketdata_message_.trade.is_packet_end = hasOrderBook ? false : true;

                if (packet_log_struct_ != nullptr)
                {
                    packet_log_struct_->Type = (int)PacketLogType::TWSE;
                    memcpy(packet_log_struct_->ChannelSeq, packet->Hdr.InformationSequenceNo, 4);
                    packet_log_struct_->LoopCount++;
                }

                Notify(std::get<1>(*t), &marketdata_message_, (void *)packet);
                index++;
            }


            if (hasOrderBook)
            {
                marketdata_message_.market_data_message_type =
                    MarketDataMessageType::MarketDataMessageType_Snapshot;

                marketdata_message_.provider_time   = ts;
                marketdata_message_.exchange_time   = Timestamp::from_epoch_nsec(0);
                marketdata_message_.sequence_number = 0;
                marketdata_message_.mbp.count       = 5;
                marketdata_message_.symbol          = std::get<0>(*t);

                int bidCount = ((*packet->InformationPrompt) & (char)TWSE_SHOW_BUYCNTMASK) >> 4;

                for (int i = 0; i < bidCount; ++i)
                {
                    marketdata_message_.mbp.bid_price[i] =
                        Decode5(packet->Records[index + i].Price);

                    marketdata_message_.mbp.bid_qty[i] = Decode4(packet->Records[index + i].Qty);
                }

                int restBidCount = 5 - bidCount;

                memset(&marketdata_message_.mbp.bid_price[bidCount], 0,
                       restBidCount * sizeof(double));

                memset(&marketdata_message_.mbp.bid_qty[bidCount], 0,
                       restBidCount * sizeof(double));

                index += bidCount;

                int askCount = ((*packet->InformationPrompt) & TWSE_SHOW_SELLCNTMASK) >> 1;
                for (int i = 0; i < askCount; ++i)
                {
                    marketdata_message_.mbp.ask_price[i] =
                        Decode5(packet->Records[index + i].Price);

                    marketdata_message_.mbp.ask_qty[i] = Decode4(packet->Records[index + i].Qty);
                }

                int restAskCount = 5 - askCount;

                memset(&marketdata_message_.mbp.ask_price[askCount], 0,
                       restAskCount * sizeof(double));

                memset(&marketdata_message_.mbp.ask_qty[askCount], 0,
                       restAskCount * sizeof(double));
                marketdata_message_.mbp.is_packet_end = true;

                if (packet_log_struct_ != nullptr)
                {
                    packet_log_struct_->Type = (int)PacketLogType::TWSE;
                    memcpy(packet_log_struct_->ChannelSeq, packet->Hdr.InformationSequenceNo, 4);
                    packet_log_struct_->LoopCount++;
                }

                Notify(std::get<1>(*t), &marketdata_message_, (void *)packet);
            }
        }
    }
}

void MarketDataProvider_TWSE::Notify(std::vector<MarketDataListener *> &list,
                                     MarketDataMessage *msg, void *raw_packet)
{
    for (auto &l : list)
    {
        l->OnMarketDataMessage(msg, raw_packet);
    }
}

const Timestamp MarketDataProvider_TWSE::PeekTimestamp()
{
    return Timestamp::invalid();
}

ProviderID MarketDataProvider_TWSE::GetProviderID() const
{
    if (data_source_id_ == DataSourceID::TSE)
    {
        return ProviderID::TSE;
    }
    else if (data_source_id_ == DataSourceID::OTC)
    {
        return ProviderID::OTC;
    }

    return ProviderID::Invalid;
}

void MarketDataProvider_TWSE::SetPacketLogStruct(PacketLogStruct *packet_log_struct)
{
    packet_log_struct_ = packet_log_struct;
    if (packet_log_struct_ != nullptr)
    {
        if (data_source_id_ == DataSourceID::TSE)
        {
            packet_log_channel_id_ = (int)PacketLogChannelID::TSE;
        }
        else if (data_source_id_ == DataSourceID::OTC)
        {
            packet_log_channel_id_ = (int)PacketLogChannelID::OTC;
        }
    }
}

PacketLogStruct *MarketDataProvider_TWSE::GetPacketLogStruct()
{
    return packet_log_struct_;
}

}  // namespace alphaone
