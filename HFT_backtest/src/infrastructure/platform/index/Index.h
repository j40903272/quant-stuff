#ifndef _INDEX_H_
#define _INDEX_H_

#include "IndexListener.h"
#include "IndexMessage.h"
#include "infrastructure/base/MarketDataListener.h"
#include "infrastructure/platform/engine/Engine.h"
#include "infrastructure/platform/manager/ObjectManager.h"

#include <unordered_map>
#include <vector>

namespace alphaone
{

class Index : public MarketDataListener
{

  public:
    Index(Engine *Engine);
    ~Index();

    // index source
    virtual void AddIndexListener(IndexListener *listener);

    // marketdata listener
    virtual void OnMarketDataMessage(const MarketDataMessage *mm, void *raw_packet);

    // utility
    virtual bool AddSymbol(const Symbol *s, double weight, long pub_shares,
                           const BookPrice &ref_price, long value, bool add_listener = true);
    virtual bool AddSymbol(const ShareInfo &sinfo, bool add_listener = true);

    // set base value for market cap index
    virtual void SetBaseIndex(double base_index);
    virtual void SetBaseMarketCap(double base_market_cap);
    virtual bool Init();


    virtual BookPrice GetPrice(BookSide side) const;
    virtual BookPrice GetTrade() const;

  protected:
    struct Component
    {
        Component(double weight, long shares, const BookPrice &ask, const BookPrice &bid)
            : weight_{weight}
            , public_shares_{shares}
            , cap_weight_{0.}
            , last_price_{bid, ask}
            , last_trade_{(bid + ask) * 0.5}
        {
        }
        double    weight_;
        long      public_shares_;
        double    cap_weight_;
        BookPrice last_price_[AskBid];
        BookPrice last_trade_;
    };

    Engine *     engine_;
    double       total_weight_;
    IndexMessage im_;
    double       base_market_cap_;
    double       curr_market_cap_[AskBid];
    double       trade_market_cap_;
    double       base_index_;
    double       base_multiplier_;

    bool init_;

    std::unordered_map<const Symbol *, Component> symbol_to_component_;
    std::vector<IndexListener *>                  listeners_;
};


}  // namespace alphaone


#endif