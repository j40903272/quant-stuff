#ifndef _LEVELFACTORY_H_
#define _LEVELFACTORY_H_

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

struct Level
{
    Level(Order *order, const BookPrice &price, const BookNord &nord, const BookSide &side);
    Level(const Timestamp &timestamp, const BookPrice &price,
          const BookSide &side);  // if there are no orders, creates blank level
    Level(const Level &) = delete;
    Level &operator=(const Level &) = delete;

    ~Level() = default;

    void Reset();
    void Set(Order *order, const BookPrice &price, const BookNord &nord, const BookSide &side);
    void Set(const Timestamp &timestamp, const BookPrice &price,
             const BookSide &side);  // blank level

    Order *   head_;  // last order (worst in queue)
    Order *   tail_;  // first order (best in queue)
    Level *   prev_;  // better price (higher bid or lower ask)
    Level *   next_;  // worse price (lower bid or higher ask)
    Timestamp time_;  // creation time of level
    BookPrice price_;
    BookQty   qty_;
    BookNord  nord_;
    BookSide  side_;
};

class LevelFactory
{
  public:
    LevelFactory();
    LevelFactory(const LevelFactory &) = delete;
    LevelFactory &operator=(const LevelFactory &) = delete;

    ~LevelFactory();

    size_t RegisterMemory(const size_t qty) const;
    // make sure to call this before any new_level or delete_level calls
    void Allocate();
    void Deallocate();

    Level *CreateLevel(Order *order, BookPrice price, BookNord nord, BookSide side);
    Level *CreateLevel(const Timestamp &timestamp, BookPrice price, BookSide side);
    void   RemoveLevel(Level *);

    bool IsAllocated() const;

  private:
    static constexpr size_t  MAX_NUMBER_OF_OBJECTS_{1024 * 4};
    SimpleMemoryPool<Level> *pool_;
    mutable size_t           size_occupied_;
};
}  // namespace alphaone

#endif
