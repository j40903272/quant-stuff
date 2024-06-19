#ifndef _PREMARKETLISTENER_H_
#define _PREMARKETLISTENER_H_

#include "infrastructure/base/PreMarket.h"
#include "infrastructure/common/message/MarketDataMessage.h"

namespace alphaone
{

class PreMarketListener
{
  public:
    PreMarketListener()                          = default;
    PreMarketListener(const PreMarketListener &) = delete;
    PreMarketListener &operator=(const PreMarketListener &) = delete;

    virtual ~PreMarketListener() = default;

    virtual void PreMarketCallBack(const Timestamp event_loop_time, const Timestamp call_back_time,
                                   void *structure) = 0;

    void TranslateMessage(const SymbolManager *symbol_manager, const PreMarketMessage *m,
                          MarketDataMessage &mm) const
    {
        if (!symbol_manager)
            return;
        memset(static_cast<void *>(&mm), 0, sizeof(mm));
        mm.symbol = symbol_manager->GetSymbolByPid("TAIFEX", m->MainType == 1 ? "FUTURE" : "OPTION",
                                                   std::string(m->Pid));
        auto dc   = mm.symbol->GetDecimalConverter();
        mm.provider_time   = Timestamp::from_epoch_nsec(m->ApTime);
        mm.exchange_time   = Timestamp::from_epoch_nsec(m->ExchTime);
        mm.sequence_number = m->SeqNo;

        if (m->DealPrice == -1)
        {
            mm.market_data_message_type = MarketDataMessageType::MarketDataMessageType_Snapshot;
            mm.mbp.count                = 5;
            for (int i = 0; i < 5; ++i)
                mm.mbp.ask_price[i] = m->Ask[i] * dc;
            for (int i = 0; i < 5; ++i)
                mm.mbp.bid_price[i] = m->Bid[i] * dc;
            for (int i = 0; i < 5; ++i)
                mm.mbp.ask_qty[i] = m->AskQty[i];
            for (int i = 0; i < 5; ++i)
                mm.mbp.bid_qty[i] = m->BidQty[i];
            mm.mbp.is_packet_end = true;
        }
        else
        {
            mm.market_data_message_type = MarketDataMessageType::MarketDataMessageType_Trade;
            mm.trade.price              = m->DealPrice * dc;
            mm.trade.qty                = m->DealQty;
            mm.trade.side               = m->Side;
            mm.trade.is_packet_end      = true;
        }
    }
};

}  // namespace alphaone
#endif
