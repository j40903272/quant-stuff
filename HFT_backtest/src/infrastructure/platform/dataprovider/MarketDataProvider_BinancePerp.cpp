#include "MarketDataProvider_BinancePerp.h"

// #include "infrastructure/common/protobuf/MarketDataMessage.pb.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/Logger.h"
#include "infrastructure/common/util/LoggerFormat.h"

#include <fcntl.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

namespace alphaone
{

MarketDataProvider_BinancePerp::MarketDataProvider_BinancePerp()
    : MarketDataProvider{alphaone::DataSourceType::MarketByPrice}
    , marketdata_message_{DataSourceType::MarketByPrice}
    , is_initialized_{false}
    , is_finished_{false}
    , reader_{nullptr}
{
    marketdata_message_.provider_id = ProviderID::BINANCE_PERP;
}

MarketDataProvider_BinancePerp::~MarketDataProvider_BinancePerp()
{
    if (reader_ != nullptr)
    {
        delete reader_;
    }
}

void MarketDataProvider_BinancePerp::Init(const std::string &file_path)
{
    file_path_ = file_path;

    reader_ = new CryptoMarketDataMessageFileReader(file_path_);
    std::cout<< "create reader: " << file_path_ << std::endl;
    is_initialized_ = true;
    if (!reader_->ReadNext()) // Read first line
        is_finished_ = true;
}

void MarketDataProvider_BinancePerp::Process(const Timestamp &event_loop_time)
{
    if (is_initialized_)
    {
        if (event_loop_time < PeekTimestamp())
        {
            return;
        }

        if (reader_->ReadNext())
        {
            SPDLOG_INFO("ReadNext la bitch");
            OnMarketData(reader_->GetMarketDataMessage());
        }
        else
        {
            is_finished_ = true;
        }
    }
    else
    {
        SPDLOG_ERROR("Process before initialization.");
        abort();
    }
}

// void MarketDataProvider_BinancePerp::OnMarketData(const protobuf::MarketDataMessage &msg)
void MarketDataProvider_BinancePerp::OnMarketData(const nlohmann::json &msg)
{
    SPDLOG_INFO("am i here?1 reader_->GetDataSourceID(): {}", reader_->GetDataSourceID());
    auto market_data_source_ = GetMarketDataSource(reader_->GetDataSourceID());
    if (market_data_source_ != nullptr)
    {
        SPDLOG_INFO("not nullptr");
        auto node{
            // market_data_source_->GetMarketDataListenerNode(msg["pid"]["c_str"], SYMBOLID_LENGTH)};
            market_data_source_->GetMarketDataListenerNode("TRBUSDT", SYMBOLID_LENGTH)};

        if (BRANCH_LIKELY(node != nullptr))
        {
            auto &symbol{std::get<0>(*node)};
            auto &listeners{std::get<1>(*node)};
            // SPDLOG_INFO("test: {}", marketdata_message_);
            marketdata_message_.type                     = GetType();
            marketdata_message_.symbol                   = symbol;

            // marketdata_message_.market_data_message_type = (MarketDataMessageType)msg["messagetype"];
            marketdata_message_.market_data_message_type = MarketDataMessageType::MarketDataMessageType_PacketEnd;
            
            // marketdata_message_.provider_time   = Timestamp::from_epoch_nsec(msg["providertime"]);
            // marketdata_message_.exchange_time   = Timestamp::from_epoch_nsec(msg["exchangetime"]);
            marketdata_message_.provider_time   = Timestamp::from_epoch_nsec((int64_t)msg["t"]);
            marketdata_message_.exchange_time   = Timestamp::from_epoch_nsec((int64_t)msg["t"]);
            SPDLOG_INFO("provider_time: {}, exchange_time: {}", marketdata_message_.provider_time, marketdata_message_.exchange_time);
            int bids_count                     = (int64_t)msg["bids"].size();
            int asks_count                     = (int64_t)msg["asks"].size();
            SPDLOG_INFO("bids_count: {}, asks_count: {}", bids_count, asks_count);
            for (int i = 0; i < bids_count; ++i)
            {
                SPDLOG_INFO("{}", msg["bids"][i]);
                marketdata_message_.mbp.bid_price[i] = std::stod(msg["bids"][i][0].get<std::string>());
                marketdata_message_.mbp.bid_qty[i]   = std::stod(msg["bids"][i][1].get<std::string>());
            }
            for (int i = 0; i < asks_count; ++i)
            {
                marketdata_message_.mbp.ask_price[i] = std::stod(msg["asks"][i][0].get<std::string>());
                marketdata_message_.mbp.ask_qty[i]   = std::stod(msg["asks"][i][1].get<std::string>());
            }
            
            // marketdata_message_.sequence_number = msg["sequencenumber"];
            marketdata_message_.sequence_number = 1;

            // if (msg.messagetype() ==
            //     protobuf::MarketDataMessage_MarketDataMessageType::
            //         MarketDataMessage_MarketDataMessageType_MarketDataMessageType_Snapshot)
            // if(true)
            // {
            //     if (msg["has_mbp"])
            //     {
            //         marketdata_message_["type"]      = DataSourceType::MarketByPrice;
            //         int count                     = msg["mbp"]["count"];
            //         marketdata_message_["mbp"]["count"] = count;
            //         for (int i = 0; i < count; ++i)
            //         {
            //             marketdata_message_.mbp.bid_price[i] = msg.mbp().bid(i);
            //             marketdata_message_.mbp.ask_price[i] = msg.mbp().ask(i);
            //             marketdata_message_.mbp.bid_qty[i]   = msg.mbp().bidqty(i);
            //             marketdata_message_.mbp.ask_qty[i]   = msg.mbp().askqty(i);
            //         }
            //         marketdata_message_.mbp.is_packet_end =
            //             msg.mbp().has_ispacketend() ? msg.mbp().ispacketend().value() : true;
            //     }
            //     else if (msg.has_mbo())
            //     {
            //         marketdata_message_.type         = DataSourceType::MarketByOrder;
            //         marketdata_message_.mbo.order_id = msg.mbo().orderid();
            //         marketdata_message_.mbo.price    = msg.mbo().price();
            //         marketdata_message_.mbo.qty      = msg.mbo().qty();
            //         marketdata_message_.mbo.nord     = msg.mbo().numberoforders();
            //         marketdata_message_.mbo.side     = msg.mbo().side();
            //         marketdata_message_.mbo.is_packet_end =
            //             msg.mbo().has_ispacketend() ? msg.mbo().ispacketend().value() : true;
            //     }
            // }
            // else if (msg.messagetype() ==
            //          protobuf::MarketDataMessage_MarketDataMessageType::
            //              MarketDataMessage_MarketDataMessageType_MarketDataMessageType_Trade)
            // {
            //     marketdata_message_.type           = DataSourceType::MarketByPrice;
            //     marketdata_message_.trade.order_id = msg.trade().orderid();

            //     marketdata_message_.trade.counterparty_order_id = msg.trade().counterpartyorderid();

            //     marketdata_message_.trade.price = msg.trade().price();
            //     marketdata_message_.trade.qty   = msg.trade().qty();
            //     marketdata_message_.trade.side  = msg.trade().side();
            //     marketdata_message_.trade.is_packet_end =
            //         msg.trade().has_ispacketend() ? msg.trade().ispacketend().value() : true;

            //     // HACK: TSE/OTC trade with duplicate seq no should be skipped
            //     if (BRANCH_UNLIKELY(msg == last_trade_))
            //     {
            //         marketdata_message_.trade.is_not_duplicate_ = false;
            //         marketdata_message_.trade.is_packet_end     = true;
            //     }
            //     else
            //     {
            //         marketdata_message_.trade.is_not_duplicate_ = true;
            //     }
            //     last_trade_ = msg;
            //     // HACK end
            // }
            // else
            // {
            //     marketdata_message_.type         = DataSourceType::MarketByOrder;
            //     marketdata_message_.mbo.order_id = msg.mbo().orderid();
            //     marketdata_message_.mbo.price    = msg.mbo().price();
            //     marketdata_message_.mbo.qty      = msg.mbo().qty();
            //     marketdata_message_.mbo.nord     = msg.mbo().numberoforders();
            //     marketdata_message_.mbo.side     = msg.mbo().side();
            //     marketdata_message_.mbo.is_packet_end =
            //         msg.mbo().has_ispacketend() ? msg.mbo().ispacketend().value() : true;
            // }

            for (auto &listener : listeners)
            {
                SPDLOG_INFO("add to listenter!");
                listener->OnMarketDataMessage(&marketdata_message_, nullptr);
            }
        }
    }
}

DataSourceType MarketDataProvider_BinancePerp::GetType() const
{
    return DataSourceType::MarketByPrice;
}

const Timestamp MarketDataProvider_BinancePerp::PeekTimestamp()
{
    // std::cout << "am i here to PeekTimestamp?" << std::endl;
    return reader_->PeekTimestamp();
}

bool MarketDataProvider_BinancePerp::IsInitialized() const
{
    return is_initialized_;
}

bool MarketDataProvider_BinancePerp::IsFinished() const
{
    return is_finished_;
}

ProviderID MarketDataProvider_BinancePerp::GetProviderID() const
{
    return ProviderID::BINANCE_PERP;
}

}  // namespace alphaone
