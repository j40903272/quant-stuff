#ifndef _TWSEFILEBOOK_H_
#define _TWSEFILEBOOK_H_

#include "infrastructure/platform/book/MarketByOrderBook.h"

namespace alphaone
{

class TWSEFileBook : public MarketByOrderBook
{
  public:
    TWSEFileBook(const Symbol *symbol, OrderFactory *order_factory, LevelFactory *level_factory);
    TWSEFileBook(const TWSEFileBook &) = delete;
    TWSEFileBook &operator=(const TWSEFileBook &) = delete;

    ~TWSEFileBook() = default;

    DataSourceType GetType() const final
    {
        return DataSourceType::MarketByOrder;
    }

    void OnAdd(const MarketDataMessage *) final;
    void OnDelete(const MarketDataMessage *) final;
    void OnModifyWithPrice(const MarketDataMessage *) final;
    void OnModifyWithQty(const MarketDataMessage *) final;
    void OnSnapshot(const MarketDataMessage *) final;
    void OnTrade(const MarketDataMessage *) final;
    void OnPacketEnd(const MarketDataMessage *) final;
    void OnSparseStop() final;

  protected:
    void RemoveOrder(Level *level, Order *order, bool operated = true) final;
    void ClearOrders(Level *level);
    std::map<ExternalOrderId, std::vector<Order *>> map_id_to_orders_;

  private:
    void EmitPreDelete() const override;
    void EmitPostAdd() const override;
    void EmitPostDelete() const override;
    void EmitPostModifyWithPrice() const override;
    void EmitPostModifyWithQty() const override;
    void EmitPostSnapshot() const override;
    void EmitPostTrade() const override;
    void EmitPacketEnd() const override;
    void EmitSparseStop() const override;

    const ExternalOrderId twse_hack_multiplier_;
    const ExternalOrderId twse_hack_space_;
    const Timestamp       twse_hack_closing_time_;
    Timestamp             twse_hack_opening_time_;
    Timestamp             last_provider_time_;
    ExternalOrderId       GetTWSEOtherOrderId(const ExternalOrderId order_id);
    void                  UncrossMarketByOrderBook(const MarketDataMessage *mm);
    BookPrice             trade_pride_;
};

}  // namespace alphaone

#endif
