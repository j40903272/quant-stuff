#ifndef _ORDERFACTORY_H_
#define _ORDERFACTORY_H_

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/memory/SimpleMemoryPool.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/Helper.h"

namespace alphaone
{
struct Order;
struct Level;
class OrderFactory;
class LevelFactory;

struct Order
{
    Order(const ExternalOrderId &id, const Timestamp &time, const BookQty &size);
    Order(const Order &) = delete;
    Order &operator=(const Order &) = delete;

    ~Order() = default;

    void Reset();
    void Set(const ExternalOrderId &id, const Timestamp &time, const BookQty &size);

    ExternalOrderId id_;
    Order *         prev_;  // better queue priority (older order)
    Order *         next_;  // worse queue priority (newer order)
    Level *         level_;
    Timestamp       time_;  // when order was created (or modified for queue priority purposes)
    BookQty         qty_;

    BookPrice GetPrice() const;
    BookSide  GetSide() const;

    std::string Dump() const
    {
        if (level_)
        {
            return fmt::format("id={} time={} qty={} price={} side={}", id_, time_, qty_,
                               GetPrice(), GetSide() == BID ? "BID" : "ASK");
        }
        else
        {
            return fmt::format("id={} time={} qty={} level=nullptr", id_, time_, qty_);
        }
    }
};

class OrderFactory
{
  public:
    OrderFactory();
    OrderFactory(const OrderFactory &) = delete;
    OrderFactory &operator=(const OrderFactory &) = delete;

    ~OrderFactory();

    size_t RegisterMemory(const size_t qty) const;
    // make sure to call this before any new_order or delete_order calls
    void Allocate();
    void Deallocate();

    Order *CreateOrder(const ExternalOrderId &id, const Timestamp &time, const BookQty &size);
    void   DeleteOrder(Order *);

    bool IsAllocated() const;

  private:
    static constexpr size_t  MAX_NUMBER_OF_OBJECTS_{1024 * 64 * 16};
    SimpleMemoryPool<Order> *pool_;
    mutable size_t           size_occupied_;
};
}  // namespace alphaone

#endif
