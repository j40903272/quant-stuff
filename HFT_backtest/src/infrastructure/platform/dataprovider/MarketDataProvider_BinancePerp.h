#ifndef _MARKET_DATA_PROVIDER_BINANCE_PERP_H
#define _MARKET_DATA_PROVIDER_BINANCE_PERP_H

#include "infrastructure/base/MarketDataListener.h"
#include "infrastructure/base/MarketDataProvider.h"
#include "infrastructure/common/file/CryptoMarketDataMessageFileReader.h"
// #include "infrastructure/common/protobuf/MarketDataMessage.pb.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/platform/datasource/MarketDataSource.h"

#include <string>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>

namespace alphaone
{
class MarketDataProvider_BinancePerp : public MarketDataProvider
{
  public:
    MarketDataProvider_BinancePerp();
    ~MarketDataProvider_BinancePerp();

    void            Init(const std::string &file_path);
    void            Process(const Timestamp &event_loop_time);
    const Timestamp PeekTimestamp();

    DataSourceType GetType() const;

    bool IsInitialized() const;
    bool IsFinished() const;

    ProviderID GetProviderID() const;

  private:
    // protobuf::MarketDataMessage message_;
    nlohmann::json message_;
    Timestamp                   message_timestamp_;

    MarketDataMessage marketdata_message_;

    std::string         file_path_;
    GlobalConfiguration configuration_;

    bool                         is_initialized_;
    bool                         is_finished_;
    CryptoMarketDataMessageFileReader *reader_;

    // void OnMarketData(const protobuf::MarketDataMessage &msg);
    void OnMarketData(const nlohmann::json &msg);
    template <typename P, typename Q>
    struct ThinTrade
    {
        ThinTrade()
        {
            price_           = 0;
            qty_             = 0;
            sequence_number_ = 0;
        }
        
        ThinTrade(const nlohmann::json &msg)
        // ThinTrade(const protobuf::MarketDataMessage &msg)
        {
            // price_           = msg.trade().price();
            // qty_             = msg.trade().qty();
            // sequence_number_ = msg.sequencenumber();
            price_           = msg["trade"]["price"];
            qty_             = msg["trade"]["qty"];
            sequence_number_ = msg["sequencenumber"];
        }
        P                  price_;
        Q                  qty_;
        int64_t            sequence_number_;
        // inline friend bool operator==(const ThinTrade &t1, const protobuf::MarketDataMessage &m1)
        inline friend bool operator==(const ThinTrade &t1, const nlohmann::json &m1)
        {
            // return (t1.sequence_number_ == m1.sequencenumber()) &&
                //    (t1.price_ == m1.trade().price()) && (t1.qty_ == m1.trade().qty());
                return (t1.sequence_number_ == m1["sequencenumber"]) &&
                   (t1.price_ == m1["trade"]["price"]) && (t1.qty_ == m1["trade"]["qty"]);
        }
        // inline friend bool operator==(const protobuf::MarketDataMessage &m1, const ThinTrade &t1)
        inline friend bool operator==(const nlohmann::json &m1, const ThinTrade &t1)
        {
            // return (t1.sequence_number_ == m1.sequencenumber()) &&
            //        (t1.price_ == m1.trade().price()) && (t1.qty_ == m1.trade().qty());
            return (t1.sequence_number_ == m1["sequencenumber"]) &&
                   (t1.price_ == m1["trade"]["price"]) && (t1.qty_ == m1["trade"]["qty"]);
                   
        }
        // inline friend bool operator!=(const ThinTrade &t1, const protobuf::MarketDataMessage &m1)
        inline friend bool operator!=(const ThinTrade &t1, const nlohmann::json &m1)
        {
            return !(t1 == m1);
        }
        // inline friend bool operator!=(const protobuf::MarketDataMessage &m1, const ThinTrade &t1)
        inline friend bool operator!=(const nlohmann::json &m1, const ThinTrade &t1)
        {
            return !(t1 == m1);
        }
    };
    ThinTrade<BookPrice, BookQty> last_trade_;
};
}  // namespace alphaone
#endif
