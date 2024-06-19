#include "MarketDataChecker.h"

#include "infrastructure/common/spdlog/spdlog.h"

#include <iostream>
namespace alphaone
{

bool operator==(const MarketDataMessageCache &c1, const MarketDataMessageCache &c2)
{
    switch (c1.mtype_)
    {
    case MarketDataMessageType_Trade:
        return (c1.symbol_ == c2.symbol_) && (c1.type_ == c2.type_) &&
               (c1.provider_time_ == c2.provider_time_) &&
               (c1.sequence_number_ == c2.sequence_number_) &&
               (c1.trade_.price == c2.trade_.price) && (c1.trade_.qty == c2.trade_.qty) &&
               (c1.trade_.order_id == c2.trade_.order_id) &&
               (c1.trade_.is_packet_end == c2.trade_.is_packet_end);
    case MarketDataMessageType_Snapshot:
        switch (c1.type_)
        {
        case DataSourceType::MarketByOrder:
            return (c1.symbol_ == c2.symbol_) && (c1.type_ == c2.type_) &&
                   (c1.provider_time_ == c2.provider_time_) &&
                   (c1.sequence_number_ == c2.sequence_number_) &&
                   (c1.mbo_.price == c2.mbo_.price) && (c1.mbo_.qty == c2.mbo_.qty) &&
                   (c1.mbo_.side == c2.mbo_.side) && (c1.mbo_.order_id == c2.mbo_.order_id) &&
                   (c1.mbo_.is_packet_end == c2.mbo_.is_packet_end);
        case DataSourceType::MarketByPrice:
            return (c1.symbol_ == c2.symbol_) && (c1.type_ == c2.type_) &&
                   (c1.provider_time_ == c2.provider_time_) &&
                   (c1.sequence_number_ == c2.sequence_number_) &&
                   (c1.mbp_.bid_price[0] == c2.mbp_.bid_price[0]) &&
                   (c1.mbp_.ask_price[0] == c2.mbp_.ask_price[0]) &&
                   (c1.mbp_.bid_qty[0] == c2.mbp_.bid_qty[0]) &&
                   (c1.mbp_.ask_qty[0] == c2.mbp_.ask_qty[0]) &&
                   (c1.mbp_.is_packet_end == c2.mbp_.is_packet_end);
        default:
            return false;
        }
    default:
        return false;
    }
}

bool operator!=(const MarketDataMessageCache &c1, const MarketDataMessageCache &c2)
{
    return !(c1 == c2);
}

MarketDataChecker::MarketDataChecker(const ObjectManager *object_manager, const bool is_light_mode)
    : MarketDataReader(object_manager)
    , json_(object_manager->GetGlobalConfiguration()->GetJson().at("Checker"))
    , squence_number_max_diff_{json_.value("sequence_number_max_diff", 50)}
    , check_layer_{json_.value("check_layer", 5) - 1}
    , message_buffer_size_{json_.value("message_buffer_size", 10)}
    , is_check_trade_distribution_{json_.value("is_check_trade_distribution", false)}
    , is_light_mode_{is_light_mode}
    , touch_price_{static_cast<BookPrice>(INT32_MIN), static_cast<BookPrice>(INT32_MAX)}
    , last_valid_message_ts_{Timestamp::invalid(), Timestamp::invalid(), Timestamp::invalid(),
                             Timestamp::invalid(), Timestamp::invalid(), Timestamp::invalid(),
                             Timestamp::invalid()}
{
    if (json_.contains("checking_symbols"))
    {
        const auto &symbol_list = json_["checking_symbols"];
        const auto &size        = static_cast<int>(symbol_list.size());
        const std::vector<std::pair<BookTrade, std::string>> str_pairs{
            {BookPart, "book_interval"}, {TradePart, "trade_interval"}};
        const std::vector<std::pair<BookTrade, double>> t_pairs{{BookPart, 5.}, {TradePart, 30.}};
        for (int symbol_id = 0; symbol_id < size; ++symbol_id)
        {
            const auto symbol_str = symbol_list[symbol_id].get<std::string>();
            const auto symbol     = GetSymbolManager()->GetSymbolByString(symbol_str);
            if (!symbol)
            {
                SPDLOG_WARN("Skip Invalid symbol_str = {}", symbol_str);
                continue;
            }
            checking_symbols_.push_back(symbol);
            checking_symbol_to_symbol_id_map_.insert({symbol, symbol_id});
            for (int i = BookPart; i < BookTradePart; ++i)
            {
                const auto str_key = str_pairs[i].second;
                intervals_[i].push_back(
                    json_.contains(str_key)
                        ? (json_[str_key].contains(symbol_str)
                               ? Duration::from_time(
                                     json_[str_key][symbol_str].get<std::string>().c_str())
                               : Duration::from_time(json_[str_key].get<std::string>().c_str()))
                        : Duration::from_sec(t_pairs[i].second));
            }
        }
    }
}

MarketDataChecker::~MarketDataChecker()
{
    const auto last_data_source_id = static_cast<size_t>(DataSourceID::END);
    for (size_t i = 0; i < last_data_source_id; ++i)
    {
        SPDLOG_CRITICAL("[{}] last_valid_message_ts_ = {}",
                        EnumToString::ToString(static_cast<DataSourceID>(i)),
                        last_valid_message_ts_[i].to_string());
    }
    const auto &symbols = checking_symbols_.size() ? checking_symbols_ : GetSymbols();
    for (int t = CheckBook; t < LastCheckingMessageType; ++t)
    {
        for (size_t i = 0; i < message_count_[t].size(); ++i)
        {
            for (size_t m = 0; m < message_count_[t][i].size(); ++m)
            {
                SPDLOG_CRITICAL("[{}] [{}] [{}] [Count] [{}]", GetProviders()[i].second,
                                symbols[m]->to_string(),
                                CheckingTypeToString(static_cast<CheckingMessageType>(t)),
                                message_count_[t][i][m]);
            }
        }
    }
    if (is_check_trade_distribution_)
    {
        for (size_t p_id = 0; p_id < price_count_.size(); ++p_id)
        {
            for (size_t s_id = 0; s_id < price_count_[p_id].size(); ++s_id)
            {
                const auto &symbol = symbols[s_id];
                for (const auto &[price, count] : price_count_[p_id][s_id])
                {
                    SPDLOG_CRITICAL("[{}] [{}] [Trades] [{}] [{}]", GetProviders()[p_id].second,
                                    symbol->to_string(), price / symbol->GetDecimalConverter(),
                                    count);
                }
            }
        }
    }
}

void MarketDataChecker::PrepareTimestampSeq()
{
    // init checking symbol structure
    const auto &provider_size = GetProviders().size();
    const auto &symbol_size =
        checking_symbols_.size() ? checking_symbols_.size() : GetSymbols().size();
    for (int t = BookPart; t < BookTradePart; ++t)
    {
        last_event_time_[t].resize(provider_size,
                                   std::vector<Timestamp>(symbol_size, Timestamp::invalid()));
    }

    for (int t = CheckBook; t < LastCheckingMessageType; ++t)
    {
        message_count_[t].resize(provider_size, std::vector<size_t>(symbol_size, 0));
    }

    // init for all providers for all checking data source
    price_count_.resize(provider_size, std::vector<std::map<BookPrice, int>>(symbol_size));
    last_seq_.resize(provider_size,
                     std::vector<int64_t>(static_cast<size_t>(DataSourceID::END), 0));
    checking_msgs_.resize(
        provider_size,
        std::vector<boost::circular_buffer<MarketDataMessageCache>>(
            symbol_size, boost::circular_buffer<MarketDataMessageCache>(message_buffer_size_)));
    SetInitFlag(true);
}

void MarketDataChecker::OnMarketDataMessage(const MarketDataMessage *mdm, void *raw_packet)
{
    // check data source sequence number
    const auto &provider_id    = GetCurrentProviderId();
    const auto  data_source_id = static_cast<size_t>(mdm->symbol->GetDataSourceID());
    const auto  last_seq{last_seq_[provider_id][data_source_id]};
    if (!is_light_mode_)
    {
        if (last_seq != 0 && mdm->sequence_number > last_seq + squence_number_max_diff_)
        {
            SPDLOG_ERROR(
                "[{}] [{}] [{}] sequence number diff too large! current seq no = {}, last seq "
                "no = {}, current_ts = {}",
                mdm->symbol->to_string(), GetProviders()[provider_id].second,
                GetMessageTypeString(mdm->market_data_message_type), mdm->sequence_number, last_seq,
                mdm->provider_time.to_string());
        }
        else if (mdm->market_data_message_type != MarketDataMessageType_TPrice &&
                 mdm->sequence_number < last_seq)
        {
            SPDLOG_ERROR(
                "[{}] [{}] [{}] sequence number reversion! current seq no = {}, last seq no = "
                "{}, current_ts = {}",
                mdm->symbol->to_string(), GetProviders()[provider_id].second,
                GetMessageTypeString(mdm->market_data_message_type), mdm->sequence_number, last_seq,
                mdm->provider_time.to_string());
        }
        last_seq_[provider_id][data_source_id] = mdm->sequence_number;
    }
    // check the data status for the symbol we want
    const auto &symbol_id = GetCheckingSymbolId(mdm->symbol);
    if (symbol_id == -1 || mdm->market_data_message_type == MarketDataMessageType_PacketEnd)
    {
        return;
    }
    const auto &symbol_name = mdm->symbol->to_string();
    if (last_valid_message_ts_[data_source_id].is_valid() &&
        mdm->provider_time < last_valid_message_ts_[data_source_id])
    {
        SPDLOG_ERROR("[{}] [] provider time reversion! current ts = {}, last ts = {}", symbol_name,
                     EnumToString::ToString(mdm->symbol->GetDataSourceID()),
                     mdm->provider_time.to_string(),
                     last_valid_message_ts_[data_source_id].to_string());
    }
    else
    {
        last_valid_message_ts_[data_source_id] = mdm->provider_time;
    }

    CheckDuplicateMsg(mdm, provider_id, symbol_id);

    switch (mdm->market_data_message_type)
    {
    case MarketDataMessageType_Snapshot:
        if (!mdm->exchange_time.is_valid())
        {
            SPDLOG_ERROR("[{}] exchange_time is invalid!", symbol_name);
        }
        if (!mdm->provider_time.is_valid())
        {
            SPDLOG_ERROR("[{}] provider_time is invalid!", symbol_name);
        }
        if (mdm->type == DataSourceType::MarketByPrice)
        {
            CheckMbpValidity(mdm);
        }
        else if (mdm->type == DataSourceType::MarketByOrder)
        {
            CheckMboValidity(mdm);
        }

        if (last_event_time_[BookPart][provider_id][symbol_id].is_valid())
        {
            if (mdm->provider_time - last_event_time_[BookPart][provider_id][symbol_id] >
                intervals_[BookPart][symbol_id])
            {
                SPDLOG_ERROR("[{}] too long no book! current ts = {}, last ts = {}", symbol_name,
                             mdm->provider_time.to_string(),
                             last_event_time_[BookPart][provider_id][symbol_id].to_string());
            }
        }
        last_event_time_[BookPart][provider_id][symbol_id] = mdm->provider_time;
        ++message_count_[CheckBook][provider_id][symbol_id];
        touch_price_[Ask] = mdm->mbp.ask_price[0];
        touch_price_[Bid] = mdm->mbp.bid_price[0];
        touch_qty_[Ask]   = mdm->mbp.ask_qty[0];
        touch_qty_[Bid]   = mdm->mbp.bid_qty[0];
        if (mdm->mbp.is_packet_end)
        {
            ++message_count_[CheckPacketEnd][provider_id][symbol_id];
        }
        break;
    case MarketDataMessageType_Trade:
    {
        if (!mdm->exchange_time.is_valid())
        {
            SPDLOG_ERROR("[{}] exchange_time is invalid!", symbol_name);
        }
        if (!mdm->provider_time.is_valid())
        {
            SPDLOG_ERROR("[{}] provider_time is invalid!", symbol_name);
        }
        CheckTradeValidity(mdm);
        if (last_event_time_[TradePart][provider_id][symbol_id].is_valid())
        {
            if (mdm->provider_time - last_event_time_[TradePart][provider_id][symbol_id] >
                intervals_[TradePart][symbol_id])
            {
                SPDLOG_ERROR("[{}] too long no trade! current ts = {}, last ts = {}", symbol_name,
                             mdm->provider_time.to_string(),
                             last_event_time_[TradePart][provider_id][symbol_id].to_string());
            }
        }
        last_event_time_[TradePart][provider_id][symbol_id] = mdm->provider_time;
        ++message_count_[CheckTrade][provider_id][symbol_id];
        if (is_check_trade_distribution_)
        {
            auto [it, is_suc] = price_count_[provider_id][symbol_id].emplace(mdm->trade.price, 1);
            if (!is_suc)
                it->second += 1;
        }
        if (mdm->trade.is_packet_end)
        {
            ++message_count_[CheckPacketEnd][provider_id][symbol_id];
        }
        break;
    }
    case MarketDataMessageType_TPrice:
    {
        if (!mdm->provider_time.is_valid())
        {
            SPDLOG_ERROR("[{}] provider_time is invalid!", symbol_name);
        }
        const auto &t = mdm->tprice;
        if (t.Price <= 0)
        {
            SPDLOG_ERROR("[{}] {} Price {} is invalid!", symbol_name, mdm->provider_time, t.Price);
        }
        if (t.BSCode != 1 && t.BSCode != -1)
        {
            SPDLOG_ERROR("[{}] {} BSCode {} is invalid!", symbol_name, mdm->provider_time,
                         t.BSCode);
        }
        last_event_time_[TradePart][provider_id][symbol_id] = mdm->provider_time;
        ++message_count_[CheckTrade][provider_id][symbol_id];
        if (t.is_packet_end)
        {
            ++message_count_[CheckPacketEnd][provider_id][symbol_id];
        }
    }
    default:
        break;
    }
}

void MarketDataChecker::CheckMbpValidity(const MarketDataMessage *mdm)
{
    for (int i = 0; i < check_layer_; ++i)
    {
        if (mdm->mbp.bid_price[i] <= mdm->mbp.bid_price[i + 1] && mdm->mbp.bid_price[i] > 0 &&
            mdm->mbp.bid_price[i + 1] > 0)
        {
            SPDLOG_ERROR("[{}] [Bid price inversion] bid_price[{}] = {}, bid_price[{}] = {}",
                         mdm->provider_time.to_string(), i, mdm->mbp.bid_price[i], i + 1,
                         mdm->mbp.bid_price[i + 1]);
        }
        if (mdm->mbp.ask_price[i] >= mdm->mbp.ask_price[i + 1] && mdm->mbp.ask_price[i] > 0 &&
            mdm->mbp.ask_price[i + 1] > 0)
        {
            SPDLOG_ERROR("[{}] [Ask price inversion] ask_price[{}] = {}, ask_price[{}] = {}",
                         mdm->provider_time.to_string(), i, mdm->mbp.ask_price[i], i + 1,
                         mdm->mbp.ask_price[i + 1]);
        }
    }
}

void MarketDataChecker::CheckMboValidity(const MarketDataMessage *mdm)
{
    // need to think of possible case
}

void MarketDataChecker::CheckTradeValidity(const MarketDataMessage *mdm)
{
    if (mdm->trade.price > touch_price_[Bid] && mdm->trade.price < touch_price_[Ask])
    {
        // trade happen in non-existing layer
        SPDLOG_WARN("[{}] [Trade between book] Trade price = {}, Trade qty = {}, Touch bid = {}, "
                    "Touch ask = {}",
                    mdm->provider_time.to_string(), mdm->trade.price, mdm->trade.qty,
                    touch_price_[Bid], touch_price_[Ask]);
    }
    else if (mdm->trade.price == touch_price_[Bid] && mdm->trade.qty > touch_qty_[Bid])
    {
        SPDLOG_WARN("[{}] [Trade bigger than book] Trade price = {}, Touch bid = {}, Trade qty = "
                    "{}. Touch bid qty = {}",
                    mdm->provider_time.to_string(), mdm->trade.price, touch_price_[Bid],
                    mdm->trade.qty, touch_qty_[Bid]);
    }
    else if (mdm->trade.price == touch_price_[Ask] && mdm->trade.qty > touch_qty_[Ask])
    {
        SPDLOG_WARN("[{}] [Trade bigger than book] Trade price = {}, Touch ask = {}, Trade qty = "
                    "{}. Touch ask qty = {}",
                    mdm->provider_time.to_string(), mdm->trade.price, touch_price_[Ask],
                    mdm->trade.qty, touch_qty_[Ask]);
    }
}

void MarketDataChecker::CheckDuplicateMsg(const MarketDataMessage *mdm, const int provider_id,
                                          const int symbol_id)
{
    MarketDataMessageCache current_msg{mdm};
    for (const auto &msg : checking_msgs_[provider_id][symbol_id])
    {
        if (current_msg == msg)
        {
            SPDLOG_ERROR("[{}] [Duplicate Message] [{}] [{}] [{}]", mdm->provider_time.to_string(),
                         mdm->symbol->to_string(),
                         GetMessageTypeString(mdm->market_data_message_type), mdm->sequence_number);
        }
    }
    checking_msgs_[provider_id][symbol_id].push_back(current_msg);
}

int MarketDataChecker::GetCheckingSymbolId(const Symbol *symbol)
{
    if (checking_symbols_.size())
    {
        if (auto it = checking_symbol_to_symbol_id_map_.find(symbol);
            it != checking_symbol_to_symbol_id_map_.end())
        {
            return it->second;
        }
        else
        {
            return -1;
        }
    }
    else
    {
        return GetSymbolId(symbol);
    }
}

std::string MarketDataChecker::GetDataSourceIDString(const DataSourceID &data_source_id)
{
    switch (data_source_id)
    {
    case DataSourceID::UNKNOWN:
        return "UNKNOWN";
    case DataSourceID::TAIFEX_FUTURE:
        return "TAIFEX_FUTURE";
    case DataSourceID::TAIFEX_OPTION:
        return "TAIFEX_OPTION";
    case DataSourceID::TSE:
        return "TSE";
    case DataSourceID::END:
        return "OTC";
    case DataSourceID::SGX_FUTURE:
        return "SGX";
    case DataSourceID::BINANCE_PERP:
        return "BINANCE_PERP";
    default:
        return "END";
    }
}

std::string MarketDataChecker::GetMessageTypeString(const MarketDataMessageType &type)
{
    switch (type)
    {
    case MarketDataMessageType_Snapshot:
        return "Snapshot";
    case MarketDataMessageType_Trade:
        return "Trade";
    default:
        return "Others";
    }
}

}  // namespace alphaone
