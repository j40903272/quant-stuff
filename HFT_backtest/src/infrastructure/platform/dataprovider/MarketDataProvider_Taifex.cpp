#include "MarketDataProvider_Taifex.h"

#include "infrastructure/common/twone/orderbook/TaifexOrderBook.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/Branch.h"
#include "infrastructure/common/util/Helper.h"
#include "infrastructure/platform/book/MarketByOrderBook.h"

#include <algorithm>

namespace alphaone
{
MarketDataProvider_Taifex::MarketDataProvider_Taifex(DataSourceID data_source_id)
    : MarketDataProvider{DataSourceType::MarketByPrice}
    , snapshot_finish_{false}
    , begin_sequence_number_{0}
    , marketdata_message_{DataSourceType::MarketByPrice}
    , data_source_id_{data_source_id}
    , packet_log_struct_{nullptr}

{
    marketdata_message_.trade.is_not_duplicate_ = true;
    timestamp_offset_                           = sizeof(timespec);
}

MarketDataProvider_Taifex::~MarketDataProvider_Taifex()
{
    auto list = market_data_source_->GetMarketDataListenerNodes();
    for (auto &t : list)
    {
        twone::TaifexOrderBook *order_book = (twone::TaifexOrderBook *)std::get<2>(t);
        if (order_book != nullptr)
        {
            delete order_book;
        }
    }
}

void MarketDataProvider_Taifex::Init()
{
    if (data_source_id_ == DataSourceID::TAIFEX_FUTURE)
    {
        transmission_code = '2';
        rb_marketdata_    = twone::RingBuffer(
            (int)twone::MARKETDATA_INDEX_RINGBUFFER_BASEINDEX::TAIFEX_FUTURE_REALTIME,
            (int)twone::MARKETDATA_BLOCK_RINGBUFFER_BASEINDEX::TAIFEX_FUTURE_REALTIME,
            MARKETDATA_TAIFEX_REALTIME_PACKET_SIZE, MARKETDATA_TAIFEX_REALTIME_RINGBUFFER_LENGTH,
            -1, 0);

        rb_marketdata_snapshot_ = twone::RingBuffer(
            (int)twone::MARKETDATA_INDEX_RINGBUFFER_BASEINDEX::TAIFEX_FUTURE_SNAPSHOT,
            (int)twone::MARKETDATA_BLOCK_RINGBUFFER_BASEINDEX::TAIFEX_FUTURE_SNAPSHOT,
            MARKETDATA_TAIFEX_SNAPSHOT_PACKET_SIZE, MARKETDATA_TAIFEX_SNAPSHOT_RINGBUFFER_LENGTH,
            -1, 0);

        rb_marketdata_rewind_ = twone::RingBuffer(
            (int)twone::MARKETDATA_INDEX_RINGBUFFER_BASEINDEX::TAIFEX_FUTURE_REWIND,
            (int)twone::MARKETDATA_BLOCK_RINGBUFFER_BASEINDEX::TAIFEX_FUTURE_REWIND,
            MARKETDATA_TAIFEX_REALTIME_PACKET_SIZE, MARKETDATA_TAIFEX_REALTIME_RINGBUFFER_LENGTH,
            -1, 0);
    }
    market_data_source_ = GetMarketDataSource(data_source_id_);
}

const Timestamp MarketDataProvider_Taifex::PeekTimestamp()
{
    SPDLOG_INFO("PeekTimestamp??W111TF");
    return Timestamp::invalid();
}

void MarketDataProvider_Taifex::Process(const Timestamp &event_loop_time)
{
    if (snapshot_finish_ == false)
    {
        if (ProcessSnapshot())
        {
            bool is_realtime_ready = false;

            //找到第一個即時封包
            int realtime_ready_index = rb_marketdata_.GetReadyIndex();

            int copy_realtime_ready_index = realtime_ready_index;
            while (realtime_ready_index >= 0)
            {
                void *data = rb_marketdata_.GetAddress(realtime_ready_index);
                twone::MARKETDATA_TAIFEX_PACKETTYPE *packetType =
                    (twone::MARKETDATA_TAIFEX_PACKETTYPE *)data;

                if (*packetType == twone::MARKETDATA_TAIFEX_PACKETTYPE::REALTIME)
                {
                    TXMarketDataHdr_RealTime_t *pHdr =
                        (TXMarketDataHdr_RealTime_t *)(((char *)data) + sizeof(int));
                    if (pHdr->TransmissionCode[0] == '0' && pHdr->MessageKind[0] == '2' &&
                        realtime_ready_index > 0)  // Reset Sequence
                    {
                        return;
                    }
                    else if (pHdr->TransmissionCode[0] == '0' && pHdr->MessageKind[0] == '2' &&
                             realtime_ready_index == 0)  // first start
                    {
                        is_realtime_ready = true;
                        rb_marketdata_.SetNextDoIndex(copy_realtime_ready_index + 1);
                        break;
                    }

                    int seqNo = Decode5(pHdr->ChannelSeq);
                    if (seqNo == begin_sequence_number_)
                    {
                        is_realtime_ready = true;
                        rb_marketdata_.SetNextDoIndex(copy_realtime_ready_index + 1);
                        break;
                    }
                }
                realtime_ready_index--;
            }

            //找到第一個Rewind封包
            int rewind_ready_index = rb_marketdata_rewind_.GetReadyIndex();

            int copy_rewind_ready_index = rewind_ready_index;
            while (rewind_ready_index >= 0)
            {
                void *data = rb_marketdata_rewind_.GetAddress(rewind_ready_index);

                TXMarketDataHdr_RealTime_t *pHdr  = (TXMarketDataHdr_RealTime_t *)data;
                int                         seqNo = Decode5(pHdr->ChannelSeq);
                if (seqNo < begin_sequence_number_)
                {
                    rb_marketdata_rewind_.SetNextDoIndex(copy_rewind_ready_index + 1);
                    break;
                }
                rewind_ready_index--;
            }

            for (int i = realtime_ready_index + 1; i <= copy_realtime_ready_index; ++i)
            {
                void *                               data = rb_marketdata_.GetAddress(i);
                twone::MARKETDATA_TAIFEX_PACKETTYPE *packetType =
                    (twone::MARKETDATA_TAIFEX_PACKETTYPE *)data;

                if (*packetType == twone::MARKETDATA_TAIFEX_PACKETTYPE::REALTIME)
                {
                    TXMarketDataHdr_RealTime_t *pHdr =
                        (TXMarketDataHdr_RealTime_t *)(((char *)data) + sizeof(int));

                    timespec *timestamp =
                        (timespec *)(((char *)data) + MARKETDATA_TAIFEX_REALTIME_PACKET_SIZE -
                                     timestamp_offset_);

                    Timestamp ts = Timestamp::from_epoch_nsec(
                        (int64_t)(timestamp->tv_sec * 1000000000 + timestamp->tv_nsec));


                    if (pHdr->TransmissionCode[0] == transmission_code &&
                        pHdr->MessageKind[0] == 'A')  // I081
                    {
                        ParseI081((TXMarketDataI081_t *)pHdr, false, false, ts);
                    }
                    else if (pHdr->TransmissionCode[0] == transmission_code &&
                             pHdr->MessageKind[0] == 'B')  // I083
                    {
                        ParseI083((TXMarketDataI083_t *)pHdr, false, false, ts);
                    }
                    else if (pHdr->TransmissionCode[0] == transmission_code &&
                             pHdr->MessageKind[0] == 'D')  // I024
                    {
                        ParseI024((TXMarketDataI024_t *)pHdr, false, false, ts);
                    }
                    else if (pHdr->TransmissionCode[0] == transmission_code &&
                             pHdr->MessageKind[0] == 'E')  // I025
                    {
                        ParseI025((TXMarketDataI025_t *)pHdr, false, false, ts);
                    }
                }
            }

            for (int i = rewind_ready_index + 1; i <= copy_rewind_ready_index; ++i)
            {
                TXMarketDataHdr_RealTime_t *pHdr =
                    (TXMarketDataHdr_RealTime_t *)rb_marketdata_rewind_.GetAddress(i);

                timespec *timestamp =
                    (timespec *)(((char *)pHdr) + MARKETDATA_TAIFEX_REALTIME_PACKET_SIZE -
                                 timestamp_offset_);

                Timestamp ts = Timestamp::from_epoch_nsec(
                    (int64_t)(timestamp->tv_sec * 1000000000 + timestamp->tv_nsec));

                if (pHdr->TransmissionCode[0] == transmission_code &&
                    pHdr->MessageKind[0] == 'A')  // I081
                {
                    ParseI081((TXMarketDataI081_t *)pHdr, false, true, ts);
                }
                else if (pHdr->TransmissionCode[0] == transmission_code &&
                         pHdr->MessageKind[0] == 'B')  // I083
                {
                    ParseI083((TXMarketDataI083_t *)pHdr, false, true, ts);
                }
                else if (pHdr->TransmissionCode[0] == transmission_code &&
                         pHdr->MessageKind[0] == 'D')  // I024
                {
                    ParseI024((TXMarketDataI024_t *)pHdr, false, true, ts);
                }
                else if (pHdr->TransmissionCode[0] == transmission_code &&
                         pHdr->MessageKind[0] == 'E')  // I025
                {
                    ParseI025((TXMarketDataI025_t *)pHdr, false, true, ts);
                }
            }

            if (is_realtime_ready == false)
            {
                return;
            }

            snapshot_finish_ = true;
        }
    }

    if (packet_log_struct_ != nullptr)
    {
        if (data_source_id_ == DataSourceID::TAIFEX_FUTURE)
        {
            packet_log_struct_->ChannelID = (int)PacketLogChannelID::FUTURE;
        }
        else if (data_source_id_ == DataSourceID::TAIFEX_OPTION)
        {
            packet_log_struct_->ChannelID = (int)PacketLogChannelID::OPTION;
        }
    }

    void *pBuff = NULL;
    while (rb_marketdata_.SequentialGet(&pBuff))
    {
        twone::MARKETDATA_TAIFEX_PACKETTYPE *packetType =
            (twone::MARKETDATA_TAIFEX_PACKETTYPE *)pBuff;

        if (*packetType == twone::MARKETDATA_TAIFEX_PACKETTYPE::FIXED)
        {
            /*TXMarketDataHdr_t *data =
                (TXMarketDataHdr_t *)(((char *)pBuff) + sizeof(int));
            if (data->TransmissionCode[0] == transmission_code &&
                data->MessageKind[0] == '2')  // I080
            {
                ParseI080((TXMarketDataI080_t *)data);
            }*/
        }
        else if (*packetType == twone::MARKETDATA_TAIFEX_PACKETTYPE::REALTIME)
        {
            TXMarketDataHdr_RealTime_t *data =
                (TXMarketDataHdr_RealTime_t *)(((char *)pBuff) + sizeof(int));

            timespec *timestamp =
                (timespec *)(((char *)pBuff) + MARKETDATA_TAIFEX_REALTIME_PACKET_SIZE -
                             timestamp_offset_);

            Timestamp ts = Timestamp::from_epoch_nsec(
                (int64_t)(timestamp->tv_sec * 1000000000 + timestamp->tv_nsec));

            if (packet_log_struct_ != nullptr)
            {
                packet_log_struct_->Type   = (int)PacketLogType::INVALID;
                packet_log_struct_->SeqNum = 101;
            }

            if (data->TransmissionCode[0] == transmission_code &&
                data->MessageKind[0] == 'A')  // I081
            {
                ParseI081((TXMarketDataI081_t *)data, true, false, ts);
            }
            else if (data->TransmissionCode[0] == transmission_code &&
                     data->MessageKind[0] == 'D')  // I024
            {
                ParseI024((TXMarketDataI024_t *)data, true, false, ts);
            }
            else if (data->TransmissionCode[0] == transmission_code &&
                     data->MessageKind[0] == 'B')  // I083
            {
                ParseI083((TXMarketDataI083_t *)data, true, false, ts);
            }
            else if (data->TransmissionCode[0] == transmission_code &&
                     data->MessageKind[0] == 'E')  // I025
            {
                ParseI025((TXMarketDataI025_t *)data, true, false, ts);
            }
            else if (data->TransmissionCode[0] == '0' &&
                     data->MessageKind[0] == '2')  // Sequence Reset
            {
                snapshot_finish_ = 0;
            }
        }
    }

    while (rb_marketdata_rewind_.SequentialGet(&pBuff))
    {
        TXMarketDataHdr_RealTime_t *data = (TXMarketDataHdr_RealTime_t *)pBuff;

        timespec *timestamp =
            (timespec *)(((char *)pBuff) + MARKETDATA_TAIFEX_REALTIME_PACKET_SIZE -
                         timestamp_offset_);

        Timestamp ts = Timestamp::from_epoch_nsec(
            (int64_t)(timestamp->tv_sec * 1000000000 + timestamp->tv_nsec));

        if (data->TransmissionCode[0] == transmission_code && data->MessageKind[0] == 'A')  // I081
        {
            ParseI081((TXMarketDataI081_t *)data, false, true, ts);
        }
        else if (data->TransmissionCode[0] == transmission_code &&
                 data->MessageKind[0] == 'D')  // I024
        {
            ParseI024((TXMarketDataI024_t *)data, false, true, ts);
        }
        else if (data->TransmissionCode[0] == transmission_code &&
                 data->MessageKind[0] == 'B')  // I083
        {
            ParseI083((TXMarketDataI083_t *)data, false, true, ts);
        }
        else if (data->TransmissionCode[0] == transmission_code &&
                 data->MessageKind[0] == 'E')  // I025
        {
            ParseI025((TXMarketDataI025_t *)data, false, true, ts);
        }
    }
}


bool MarketDataProvider_Taifex::ProcessSnapshot()
{
    int  snapshot_ready_index      = rb_marketdata_snapshot_.GetReadyIndex();
    int  copy_snapshot_ready_index = snapshot_ready_index;
    bool find                      = false;

    while (snapshot_ready_index >= 0)
    {
        TXMarketDataI084_t *i084 =
            (TXMarketDataI084_t *)rb_marketdata_snapshot_.GetAddress(snapshot_ready_index);
        if (i084->MsgType[0] == 'A')
        {
            find = true;
            break;
        }
        snapshot_ready_index--;
    }

    if (find)
    {
        for (int s = snapshot_ready_index; s <= copy_snapshot_ready_index; ++s)
        {
            void *                      data = rb_marketdata_snapshot_.GetAddress(s);
            TXMarketDataHdr_RealTime_t *pHdr = (TXMarketDataHdr_RealTime_t *)data;
            if (pHdr->TransmissionCode[0] == transmission_code &&
                pHdr->MessageKind[0] == 'C')  // I084
            {
                TXMarketDataI084_t *pI084 = (TXMarketDataI084_t *)data;
                if (pI084->MsgType[0] == 'O')
                {
                    TXMarketDataI084_O_Header_t *pI084_O_Header =
                        (TXMarketDataI084_O_Header_t *)((char *)data + sizeof(TXMarketDataI084_t));
                    int no_entrys = Decode1(pI084_O_Header->No_Entries);
                    int offset    = 0;
                    for (int i = 0; i < no_entrys; ++i)
                    {
                        TXMarketDataI084_O_Product_t *i084_product_ =
                            (TXMarketDataI084_O_Product_t *)((char *)pI084_O_Header +
                                                             sizeof(TXMarketDataI084_O_Header_t) +
                                                             offset);

                        int no_md_entrys = Decode1(i084_product_->NO_MD_ENTRIES);

                        auto t = market_data_source_->GetMarketDataListenerNode(
                            i084_product_->ProductID, SYMBOLID_LENGTH);

                        if (t != nullptr)
                        {
                            auto taifex_orderbook = (twone::TaifexOrderBook *)std::get<2>(*t);
                            int  seq              = Decode5(i084_product_->Last_Prod_Msg_Seq);
                            if (BRANCH_LIKELY(taifex_orderbook != nullptr))
                            {
                                taifex_orderbook->Reset();
                                int infoOffset = 0;
                                for (int j = 0; j < no_md_entrys; ++j)
                                {
                                    TXMarketDataI084_O_Info_t *pInfo =
                                        (TXMarketDataI084_O_Info_t
                                             *)((char *)i084_product_ +
                                                sizeof(TXMarketDataI084_O_Product_t) + infoOffset);

                                    taifex_orderbook->ParseI084(pInfo, seq);
                                    infoOffset += sizeof(TXMarketDataI084_O_Info_t);
                                }
                            }
                        }
                        offset += sizeof(TXMarketDataI084_O_Product_t);
                        offset += no_md_entrys * sizeof(TXMarketDataI084_O_Info_t);
                    }
                }
                else if (pI084->MsgType[0] == 'A')
                {
                    TXMarketDataI084_A_t *pI084_A =
                        (TXMarketDataI084_A_t *)((char *)data + sizeof(TXMarketDataI084_t));
                    begin_sequence_number_ = Decode5(pI084_A->Last_Seq);
                }
            }
        }
        return true;
    }
    return false;
}


void MarketDataProvider_Taifex::ParseI081(TXMarketDataI081_t *pI081, bool notify, bool isrewind,
                                          Timestamp &ts)
{
    auto t = market_data_source_->GetMarketDataListenerNode(pI081->ProductID, SYMBOLID_LENGTH);
    if (BRANCH_LIKELY(t != nullptr))
    {
        auto taifex_orderbook = (twone::TaifexOrderBook *)std::get<2>(*t);

        twone::TaifexUpdateFlag flag = taifex_orderbook->ParseI081(pI081, isrewind, ts);

        if (notify && ((uint8_t)(flag & twone::TaifexUpdateFlag::OrderBook) != 0))
        {
            taifex_orderbook->msg.market_data_message_type =
                alphaone::MarketDataMessageType::MarketDataMessageType_Snapshot;
            if (packet_log_struct_ != nullptr)
            {
                packet_log_struct_->Type = (int)PacketLogType::TAIFEX;
                memcpy(packet_log_struct_->ChannelSeq, pI081->Hdr.ChannelSeq, 5);
                packet_log_struct_->LoopCount++;
            }
            Notify(std::get<1>(*t), &taifex_orderbook->msg, (void *)pI081);
        }

        if (notify && ((uint8_t)(flag & twone::TaifexUpdateFlag::ImpliedBook) != 0))
        {
            taifex_orderbook->msg.market_data_message_type =
                alphaone::MarketDataMessageType::MarketDataMessageType_Implied;

            Notify(std::get<1>(*t), &taifex_orderbook->msg, (void *)pI081);
        }
    }
}

void MarketDataProvider_Taifex::ParseI083(TXMarketDataI083_t *pI083, bool notify, bool isrewind,
                                          Timestamp &ts)
{
    auto t = market_data_source_->GetMarketDataListenerNode(pI083->ProductID, SYMBOLID_LENGTH);

    if (BRANCH_LIKELY(t != nullptr))
    {
        auto taifex_orderbook = (twone::TaifexOrderBook *)std::get<2>(*t);

        twone::TaifexUpdateFlag flag = taifex_orderbook->ParseI083(pI083, isrewind, ts);
        if (pI083->Calculated_Flag[0] == '0' && notify)  //第一盤
        {
            if (((uint8_t)(flag & twone::TaifexUpdateFlag::OrderBook) != 0))
            {
                taifex_orderbook->msg.market_data_message_type =
                    alphaone::MarketDataMessageType::MarketDataMessageType_Snapshot;

                Notify(std::get<1>(*t), &taifex_orderbook->msg, (void *)pI083);
            }

            if ((uint8_t)(flag & twone::TaifexUpdateFlag::ImpliedBook) != 0)
            {
                taifex_orderbook->msg.market_data_message_type =
                    alphaone::MarketDataMessageType::MarketDataMessageType_Implied;

                if (packet_log_struct_ != nullptr)
                {
                    packet_log_struct_->Type = (int)PacketLogType::TAIFEX;
                    memcpy(packet_log_struct_->ChannelSeq, pI083->Hdr.ChannelSeq, 5);
                    packet_log_struct_->LoopCount++;
                }

                Notify(std::get<1>(*t), &taifex_orderbook->msg, (void *)pI083);
            }
        }
    }
}

void MarketDataProvider_Taifex::ParseI024(TXMarketDataI024_t *pI024, bool notify, bool isrewind,
                                          Timestamp &ts)
{
    auto t = market_data_source_->GetMarketDataListenerNode(pI024->ProductID, SYMBOLID_LENGTH);

    if (BRANCH_LIKELY(t != nullptr))
    {
        auto taifex_orderbook = (twone::TaifexOrderBook *)std::get<2>(*t);
        taifex_orderbook->ParseI024(pI024, isrewind, ts);

        if (notify)
        {
            marketdata_message_.market_data_message_type =
                MarketDataMessageType::MarketDataMessageType_Trade;
            marketdata_message_.symbol                      = std::get<0>(*t);
            marketdata_message_.provider_time               = ts;
            marketdata_message_.exchange_time               = Timestamp::from_epoch_nsec(0);
            marketdata_message_.sequence_number             = 0;
            marketdata_message_.trade.counterparty_order_id = 0;
            marketdata_message_.trade.order_id              = 0;

            int sign1 = pI024->FirstMatchInfo.Sign[0] == '0' ? 1 : -1;

            marketdata_message_.trade.price = Decode5(&pI024->FirstMatchInfo.Price[0]) * sign1;
            marketdata_message_.trade.qty   = Decode4(&pI024->FirstMatchInfo.Qty[0]);
            marketdata_message_.trade.side  = 0;

            int matchNumber = pI024->FurtherMatchItemNo[0] & 0x7F;

            if (matchNumber == 0)
            {
                marketdata_message_.trade.is_packet_end = true;
            }
            else
            {
                marketdata_message_.trade.is_packet_end = false;
            }

            if (packet_log_struct_ != nullptr)
            {
                packet_log_struct_->Type = (int)PacketLogType::TAIFEX;
                memcpy(packet_log_struct_->ChannelSeq, pI024->Hdr.ChannelSeq, 5);
                packet_log_struct_->LoopCount++;
            }

            Notify(std::get<1>(*t), &marketdata_message_, (void *)pI024);


            if (matchNumber > 0)
            {
                for (int i = 0; i < matchNumber; ++i)
                {
                    if (i == matchNumber - 1)
                    {
                        marketdata_message_.trade.is_packet_end = true;
                    }

                    int sign2 = pI024->FurtherMatchInfo[i].Sign[0] == '0' ? 1 : -1;

                    marketdata_message_.trade.price =
                        Decode5(&pI024->FurtherMatchInfo[i].Price[0]) * sign2;
                    marketdata_message_.trade.qty = Decode2(&pI024->FurtherMatchInfo[i].Qty[0]);

                    if (packet_log_struct_ != nullptr)
                    {
                        packet_log_struct_->Type = (int)PacketLogType::TAIFEX;
                        memcpy(packet_log_struct_->ChannelSeq, pI024->Hdr.ChannelSeq, 5);
                        packet_log_struct_->LoopCount++;
                        packet_log_struct_->SeqNum += 100;
                    }

                    Notify(std::get<1>(*t), &marketdata_message_, (void *)pI024);
                }
            }
        }
    }
}

void MarketDataProvider_Taifex::ParseI025(TXMarketDataI025_t *pI025, bool notify, bool isrewind,
                                          Timestamp &ts)
{
    auto t = market_data_source_->GetMarketDataListenerNode(pI025->ProductID, SYMBOLID_LENGTH);

    if (BRANCH_LIKELY(t != nullptr))
    {
        auto taifex_orderbook = (twone::TaifexOrderBook *)std::get<2>(*t);
        taifex_orderbook->ParseI025(pI025, isrewind, ts);
    }
}

void MarketDataProvider_Taifex::ParseI080(TXMarketDataI080_t *pI080)
{
    /*auto t = market_data_source_->GetMarketDataListenerNode(pI080->ProductID, SYMBOLID_LENGTH);

    if (BRANCH_LIKELY(t != nullptr))
    {
        marketdata_message_.market_data_message_type =
            MarketDataMessageType::MarketDataMessageType_Snapshot;

        marketdata_message_.provider_time   = Timestamp::now();
        marketdata_message_.exchange_time   = Timestamp::from_epoch_nsec(0);
        marketdata_message_.sequence_number = 0;

        marketdata_message_.mbp.count        = 5;
        marketdata_message_.symbol           = std::get<0>(*t);
        marketdata_message_.mbp.bid_price[0] = Decode5(pI080->BuyOrder1.Price);
        marketdata_message_.mbp.bid_price[1] = Decode5(pI080->BuyOrder2.Price);
        marketdata_message_.mbp.bid_price[2] = Decode5(pI080->BuyOrder3.Price);
        marketdata_message_.mbp.bid_price[3] = Decode5(pI080->BuyOrder4.Price);
        marketdata_message_.mbp.bid_price[4] = Decode5(pI080->BuyOrder5.Price);

        marketdata_message_.mbp.bid_qty[0] = Decode4(pI080->BuyOrder1.Qty);
        marketdata_message_.mbp.bid_qty[1] = Decode4(pI080->BuyOrder2.Qty);
        marketdata_message_.mbp.bid_qty[2] = Decode4(pI080->BuyOrder3.Qty);
        marketdata_message_.mbp.bid_qty[3] = Decode4(pI080->BuyOrder4.Qty);
        marketdata_message_.mbp.bid_qty[4] = Decode4(pI080->BuyOrder5.Qty);

        marketdata_message_.mbp.ask_price[0] = Decode5(pI080->SellOrder1.Price);
        marketdata_message_.mbp.ask_price[1] = Decode5(pI080->SellOrder2.Price);
        marketdata_message_.mbp.ask_price[2] = Decode5(pI080->SellOrder3.Price);
        marketdata_message_.mbp.ask_price[3] = Decode5(pI080->SellOrder4.Price);
        marketdata_message_.mbp.ask_price[4] = Decode5(pI080->SellOrder5.Price);

        marketdata_message_.mbp.ask_qty[0] = Decode4(pI080->SellOrder1.Qty);
        marketdata_message_.mbp.ask_qty[1] = Decode4(pI080->SellOrder2.Qty);
        marketdata_message_.mbp.ask_qty[2] = Decode4(pI080->SellOrder3.Qty);
        marketdata_message_.mbp.ask_qty[3] = Decode4(pI080->SellOrder4.Qty);
        marketdata_message_.mbp.ask_qty[4] = Decode4(pI080->SellOrder5.Qty);

        Notify(std::get<1>(*t), &marketdata_message_, (void *)pI080);

        marketdata_message_.market_data_message_type =
            MarketDataMessageType::MarketDataMessageType_PacketEnd;
        Notify(std::get<1>(*t), &marketdata_message_, nullptr);
    }*/
}

void MarketDataProvider_Taifex::Notify(std::vector<MarketDataListener *> &list,
                                       MarketDataMessage *msg, void *raw_packet)
{
    for (auto &l : list)
    {
        l->OnMarketDataMessage(msg, raw_packet);
    }
}

ProviderID MarketDataProvider_Taifex::GetProviderID() const
{
    if (data_source_id_ == DataSourceID::TAIFEX_FUTURE)
    {
        return ProviderID::TAIFEX_FUTURE;
    }
    else if (data_source_id_ == DataSourceID::TAIFEX_OPTION)
    {
        return ProviderID::TAIFEX_OPTION;
    }
    return ProviderID::Invalid;
}

void MarketDataProvider_Taifex::AddMarketDataListener(const Symbol *      symbol,
                                                      MarketDataListener *listener)
{
    MarketDataProvider::AddMarketDataListener(symbol, listener);
    twone::TaifexOrderBook *order_book = new twone::TaifexOrderBook(symbol);
    if (!AddExtraData(symbol, order_book))
    {
        delete order_book;
    }
}

void MarketDataProvider_Taifex::SetPacketLogStruct(PacketLogStruct *packet_log_struct)
{
    packet_log_struct_ = packet_log_struct;
}

PacketLogStruct *MarketDataProvider_Taifex::GetPacketLogStruct()
{
    return packet_log_struct_;
}

}  // namespace alphaone
