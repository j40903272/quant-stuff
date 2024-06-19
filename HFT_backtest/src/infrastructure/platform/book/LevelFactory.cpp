#include "LevelFactory.h"

#include "infrastructure/platform/book/OrderFactory.h"

namespace alphaone
{
Level::Level(Order *order, const BookPrice &price, const BookNord &nord, const BookSide &side)
    : head_{order}
    , tail_{order}
    , prev_{nullptr}
    , next_{nullptr}
    , time_{order->time_}
    , price_{price}
    , qty_{order->qty_}
    , nord_{nord}
    , side_{side}
{
}

Level::Level(const Timestamp &timestamp, const BookPrice &price, const BookSide &side)
    : head_{nullptr}
    , tail_{nullptr}
    , prev_{nullptr}
    , next_{nullptr}
    , time_{timestamp}
    , price_{price}
    , qty_{0}
    , nord_{0}
    , side_{side}
{
}

void Level::Reset()
{
    prev_ = nullptr;
    next_ = nullptr;
}

void Level::Set(Order *order, const BookPrice &price, const BookNord &nord, const BookSide &side)
{
    head_ = tail_ = order;
    time_         = order->time_;
    qty_          = order->qty_;
    price_        = price;
    nord_         = nord;
    side_         = side;
}

void Level::Set(const Timestamp &timestamp, const BookPrice &price, const BookSide &side)
{
    head_ = tail_ = nullptr;
    time_         = timestamp;
    qty_          = 0;
    price_        = price;
    nord_         = 0;
    side_         = side;
}

LevelFactory::LevelFactory() : pool_{nullptr}, size_occupied_{0}
{
}

LevelFactory::~LevelFactory()
{
    Deallocate();
}

bool LevelFactory::IsAllocated() const
{
    return pool_ != nullptr;
}

size_t LevelFactory::RegisterMemory(const size_t qty) const
{
    const size_t offset{sizeof(Level) + size_occupied_};
    size_occupied_ += qty;
    return offset;
}

void LevelFactory::Allocate()
{
    pool_ = new SimpleMemoryPool<Level>(MAX_NUMBER_OF_OBJECTS_, size_occupied_);
    for (size_t i{0}; i < MAX_NUMBER_OF_OBJECTS_; ++i)
    {
        Level *level{(Level *)pool_->ptr(i)};
        level->Reset();
    }
}

void LevelFactory::Deallocate()
{
    delete pool_;

    pool_          = nullptr;
    size_occupied_ = 0;
}

Level *LevelFactory::CreateLevel(Order *order, BookPrice price, BookNord nord, BookSide side)
{
    Level *level;
    if (BRANCH_UNLIKELY(pool_->IsEmpty()))
    {
        level = (Level *)pool_->InsufficientMalloc();
        new (level) Level(order, price, nord, side);
    }
    else
    {
        level = (Level *)pool_->SufficientMalloc();
        level->Set(order, price, nord, side);
    }
    return level;
}

Level *LevelFactory::CreateLevel(const Timestamp &timestamp, BookPrice price, BookSide side)
{
    Level *level;
    if (BRANCH_UNLIKELY(pool_->IsEmpty()))
    {
        level = ((Level *)pool_->InsufficientMalloc());
        new (level) Level(timestamp, price, side);
    }
    else
    {
        level = (Level *)pool_->SufficientMalloc();
        level->Set(timestamp, price, side);
    }
    return level;
}

void LevelFactory::RemoveLevel(Level *level)
{
    if (BRANCH_LIKELY(pool_->IsInPool(level)))
    {
        level->Reset();
        pool_->SufficientFree(level);
    }
    else
    {
        pool_->InsufficientFree(level);
    }
}
}  // namespace alphaone
