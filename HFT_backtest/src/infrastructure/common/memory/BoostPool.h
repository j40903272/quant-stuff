
#ifndef _BOOSTPOOL_H
#define _BOOSTPOOL_H
#include "boost/pool/pool.hpp"

template <typename T>
T *GetObjFromPool(boost::pool<boost::default_user_allocator_malloc_free> *pool)
{
    auto r = static_cast<T *>(pool->malloc());
    new (r) T();
    return r;
}

template <typename T, class... Us>
T *GetObjFromPool(boost::pool<boost::default_user_allocator_malloc_free> *pool, const Us... args)
{
    auto r = static_cast<T *>(pool->malloc());
    new (r) T(args...);
    return r;
}

template <typename T>
void DestroyObjFromPool(T *t_ptr, boost::pool<boost::default_user_allocator_malloc_free> *pool)
{
    // may need to check whether this is needed to speed up
    if (!std::is_integral<T>::value)
    {
        t_ptr->~T();
    }
    pool->free(t_ptr);
}

#endif