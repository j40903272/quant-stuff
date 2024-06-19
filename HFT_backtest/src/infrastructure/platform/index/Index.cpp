#include "Index.h"

namespace alphaone
{


Index::Index(Engine *engine)
    : engine_{engine}
    , total_weight_{0.}
    , im_{{0., 0.}, 0., nullptr, MarketDataMessageType_Invalid}
    , base_market_cap_{1.}
    , curr_market_cap_{0., 0.}
    , trade_market_cap_{0.}
    , base_index_{1.}
    , base_multiplier_{0.}
    , init_{false}
{
    if (!engine_)
        throw std::runtime_error("Engine is nullptr");
}

Index::~Index()
{
}

void Index::AddIndexListener(IndexListener *listener)
{
    if (std::find(listeners_.begin(), listeners_.end(), listener) == listeners_.end())
        listeners_.push_back(listener);
}

void Index::OnMarketDataMessage(const MarketDataMessage *mm, void *raw_packet)
{
    if (VERY_UNLIKELY(!init_))
        return;

    auto it = symbol_to_component_.find(mm->symbol);
    if (it == symbol_to_component_.end())
        return;

    auto &component = it->second;

    switch (mm->market_data_message_type)
    {
    case MarketDataMessageType_Snapshot:
    {
        const auto &new_ask  = mm->GetMbpAskPrice(0) == 0.
                                   ? (mm->GetMbpAskQty(0) == 0. ? mm->symbol->GetLimit<Up>()[0]
                                                                : mm->symbol->GetLimit<Down>()[0])
                                   : mm->GetMbpAskPrice(0);
        const auto &new_bid  = mm->GetMbpBidPrice(0) == 0.
                                   ? (mm->GetMbpBidQty(0) == 0. ? mm->symbol->GetLimit<Down>()[0]
                                                                : mm->symbol->GetLimit<Up>()[0])
                                   : mm->GetMbpBidPrice(0);
        const auto &ask_diff = new_ask - component.last_price_[Ask];
        const auto &bid_diff = new_bid - component.last_price_[Bid];
        if (!ask_diff && !bid_diff)
            return;

        curr_market_cap_[Ask] += ask_diff * component.public_shares_;
        curr_market_cap_[Bid] += bid_diff * component.public_shares_;
        component.last_price_[Ask] = new_ask;
        component.last_price_[Bid] = new_bid;

        if (component.last_trade_ > component.last_price_[Ask] ||
            component.last_trade_ < component.last_price_[Bid])
        {
            const auto &new_trade = (component.last_price_[Ask] + component.last_price_[Bid]) * 0.5;
            const auto &trade_diff = new_trade - component.last_trade_;
            trade_market_cap_ += trade_diff * component.public_shares_;
            component.last_trade_ = new_trade;
            im_.trade_            = trade_market_cap_ * base_multiplier_;
        }

        im_.price_[Ask] = curr_market_cap_[Ask] * base_multiplier_;
        im_.price_[Bid] = curr_market_cap_[Bid] * base_multiplier_;
    }
    break;
    case MarketDataMessageType_Trade:
    {
        const auto &new_trade =
            mm->GetTradePrice() <= component.last_price_[Ask] &&
                    mm->GetTradePrice() >= component.last_price_[Bid]
                ? mm->GetTradePrice()
                : (component.last_price_[Ask] + component.last_price_[Bid]) * 0.5;
        const auto &trade_diff = new_trade - component.last_trade_;
        if (!trade_diff)
            return;

        trade_market_cap_ += trade_diff * component.public_shares_;
        component.last_trade_ = new_trade;
        im_.trade_            = trade_market_cap_ * base_multiplier_;
    }
    break;
    default:
        return;
    }

    im_.symbol_ = mm->symbol;
    im_.type_   = mm->market_data_message_type;

    for (auto &listener : listeners_)
        listener->OnUpdate(mm->provider_time, &im_);
}

bool Index::AddSymbol(const Symbol *s, double weight, long pub_shares, const BookPrice &ref_price,
                      long value, bool add_listener)
{
    // stop trading symbol
    if (!s)
    {
        curr_market_cap_[Ask] += value;
        curr_market_cap_[Bid] += value;
        trade_market_cap_ += value;
        total_weight_ += weight;
        return true;
    }

    auto [it, is_success] =
        symbol_to_component_.emplace(s, Component{weight, pub_shares, ref_price, ref_price});
    if (is_success)
    {
        const auto &v = pub_shares * ref_price;
        curr_market_cap_[Ask] += v;
        curr_market_cap_[Bid] += v;
        trade_market_cap_ += v;
        total_weight_ += weight;
        if (add_listener)
            engine_->AddMarketDataListener(s, this);
    }
    return is_success;
}

bool Index::AddSymbol(const ShareInfo &sinfo, bool add_listener)
{
    return AddSymbol(sinfo.GetSymbol(), sinfo.GetRefWeight(), sinfo.GetNextPublicShares(),
                     sinfo.GetRefOpeningPrice(), sinfo.GetNextMarketCap(), add_listener);
}

bool Index::Init()
{
    base_multiplier_ = 1 / base_market_cap_ * base_index_;
    im_.price_[Ask]  = curr_market_cap_[Ask] * base_multiplier_;
    im_.price_[Bid]  = curr_market_cap_[Bid] * base_multiplier_;
    im_.trade_       = im_.price_[Bid];

    if (im_.price_[Ask] && im_.price_[Bid])
    {
        init_ = std::abs(total_weight_ - 1.0) <= 1e-2;
        return init_;
    }
    return false;
}

void Index::SetBaseIndex(double base_index)
{
    base_index_ = base_index;
}

void Index::SetBaseMarketCap(double base_market_cap)
{
    base_market_cap_ = base_market_cap;
}

BookPrice Index::GetPrice(BookSide side) const
{
    return im_.price_[side];
}

BookPrice Index::GetTrade() const
{
    return im_.trade_;
}

}  // namespace alphaone
