#include "OrderFactory.h"

#include "infrastructure/platform/book/LevelFactory.h"

namespace alphaone
{
Order::Order(const ExternalOrderId &id, const Timestamp &time, const BookQty &qty)
    : id_{id}, prev_{nullptr}, next_{nullptr}, level_{nullptr}, time_{time}, qty_{qty}
{
}

void Order::Reset()
{
    prev_  = nullptr;
    next_  = nullptr;
    level_ = nullptr;
}

void Order::Set(const ExternalOrderId &id, const Timestamp &time, const BookQty &qty)
{
    id_   = id;
    time_ = time;
    qty_  = qty;
}

BookPrice Order::GetPrice() const
{
    return level_->price_;
}

BookSide Order::GetSide() const
{
    return level_->side_;
}

OrderFactory::OrderFactory() : pool_{nullptr}, size_occupied_{0}
{
}

OrderFactory::~OrderFactory()
{
    Deallocate();
}

bool OrderFactory::IsAllocated() const
{
    return pool_ != nullptr;
}

size_t OrderFactory::RegisterMemory(const size_t qty) const
{
    const size_t offset{sizeof(Order) + size_occupied_};
    size_occupied_ += qty;
    return offset;
}

void OrderFactory::Allocate()
{
    pool_ = new SimpleMemoryPool<Order>{MAX_NUMBER_OF_OBJECTS_, size_occupied_};
    for (size_t i{0}; i < MAX_NUMBER_OF_OBJECTS_; ++i)
    {
        Order *order{(Order *)pool_->ptr(i)};
        order->id_ = 0;
        order->Reset();
    }
}

void OrderFactory::Deallocate()
{
    delete pool_;

    pool_          = nullptr;
    size_occupied_ = 0;
}

Order *OrderFactory::CreateOrder(const ExternalOrderId &id, const Timestamp &time,
                                 const BookQty &qty)
{
    Order *order;
    if (BRANCH_UNLIKELY(pool_->IsEmpty()))
    {
        order = ((Order *)pool_->InsufficientMalloc());
        new (order) Order(id, time, qty);
    }
    else
    {
        order = (Order *)pool_->SufficientMalloc();
        order->Set(id, time, qty);
    }
    return order;
}

void OrderFactory::DeleteOrder(Order *order)
{
    if (BRANCH_LIKELY(pool_->IsInPool(order)))
    {
        order->Reset();
        pool_->SufficientFree(order);
    }
    else
    {
        pool_->InsufficientFree(order);
    }
}
}  // namespace alphaone
