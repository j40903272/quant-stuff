#ifndef SIMPLEMEMORYPOOL_H
#define SIMPLEMEMORYPOOL_H

#include "infrastructure/common/util/Branch.h"

#include <cstdlib>
#include <cstring>

namespace alphaone
{
// SimpleMemoryPool is a class that can be used to speed up memoey alloaction when new and delete.
// Since our tasks are relatively simple, we assume that SimpleMemoryPoll handles memory allocation
// for "only one certain data type (defined by typename T)".
template <typename T>
class SimpleMemoryPool
{
  public:
    SimpleMemoryPool(const size_t max_number_of_objects, const size_t size_buffer_ = 0);
    ~SimpleMemoryPool();

    /*
    README (2020-02-19) [Andrew Kuo]
    ├─  Use those two well-encapsulated methods unless otherwise specified or required
    │   ├─  void *Malloc();
    │   └─  void Free(void *p);
    └─  Use those methods only when delicated performance improvements are required
        ├─  void *InsufficientMalloc();
        ├─  void* SufficientMalloc();
        ├─  void InsufficientFree(void *p);
        ├─  void SufficientFree(void *p);
        ├─  void* ptr(const std::size_t index);
        ├─  bool IsEmpty() const;
        └─  bool IsInPool(void* p) const;
    */

    inline void *Malloc()
    {
        if (BRANCH_LIKELY(back_index_ != -1))
        {
            return SufficientMalloc();
        }
        else
        {
            return InsufficientMalloc();
        }
    }

    inline void Free(void *p)
    {
        const char *cp{reinterpret_cast<const char *>(p)};
        if (BRANCH_LIKELY(cp >= pool_ && cp < pool_ + POOL_SIZE_))
        {
            SufficientFree(p);
        }
        else
        {
            InsufficientFree(p);
        }
    }

    inline void *InsufficientMalloc()
    {
        return new char[OBJECT_SIZE_];
    }

    inline void InsufficientFree(void *p)
    {
        delete ((T *)p);
    }

    inline void *SufficientMalloc()
    {
        return free_items_[back_index_--];
    }

    inline void SufficientFree(void *p)
    {
        free_items_[++back_index_] = p;
    }

    inline void *ptr(const size_t index)
    {
        return pool_ + (index * OBJECT_SIZE_);
    }

    inline bool IsEmpty() const
    {
        return back_index_ == -1;
    }

    inline bool IsInPool(void *p) const
    {
        const char *cp = reinterpret_cast<const char *>(p);
        return cp >= pool_ && cp < pool_ + POOL_SIZE_;
    }

  private:
    const size_t OBJECT_SIZE_;
    const size_t MAX_NUMBER_OF_OBJECTS_;
    const size_t POOL_SIZE_;

    char *        pool_;
    void **       free_items_;
    long long int back_index_;
};

template <typename T>
SimpleMemoryPool<T>::SimpleMemoryPool(const size_t max_number_of_objects, const size_t size_buffer_)
    : OBJECT_SIZE_{sizeof(T) + size_buffer_}
    , MAX_NUMBER_OF_OBJECTS_{max_number_of_objects}
    , POOL_SIZE_{OBJECT_SIZE_ * MAX_NUMBER_OF_OBJECTS_}
    , back_index_{static_cast<long long int>(max_number_of_objects - 1)}
{
    pool_       = new char[POOL_SIZE_];
    free_items_ = new void *[MAX_NUMBER_OF_OBJECTS_];
    for (size_t i{0}; i < MAX_NUMBER_OF_OBJECTS_; ++i)
    {
        free_items_[i] = pool_ + (i * OBJECT_SIZE_);
    }
    memset(pool_, 0, sizeof(char) * POOL_SIZE_);
}

template <typename T>
SimpleMemoryPool<T>::~SimpleMemoryPool()
{
    delete[] pool_;
    delete[] free_items_;
}

}  // namespace alphaone

#endif
