#ifndef _SIDE_H_
#define _SIDE_H_

#include "infrastructure/common/typedef/Typedefs.h"

namespace alphaone
{
struct BidType;
struct AskType;

struct BidType
{
    using other = AskType;
    constexpr static bool GetSide()
    {
        return BID;
    }

    constexpr static BidType *GetType()
    {
        return (BidType *)nullptr;
    }

    constexpr static AskType *GetOtherType()
    {
        return (AskType *)nullptr;
    }

    constexpr static int GetSign()
    {
        return 1;
    }

    template <typename T>
    static T AddAway(T lhs, T rhs)
    {
        return lhs - rhs;
    }

    template <typename T>
    static T AddInner(T lhs, T rhs)
    {
        return lhs + rhs;
    }

    template <typename T>
    static bool IsInner(T lhs, T rhs)
    {
        return lhs > rhs;
    }

    template <typename T>
    static bool IsInnerOrEqual(T lhs, T rhs)
    {
        return lhs >= rhs;
    }

    template <typename T>
    static T IsCrossed(T lhs, T rhs)
    {
        return lhs <= rhs;
    }

    template <typename T>
    static T GetInner(T lhs, T rhs)
    {
        return std::max(lhs, rhs);
    }

    template <typename T>
    static T GetOuter(T lhs, T rhs)
    {
        return std::min(lhs, rhs);
    }
};

struct AskType
{
    using other = BidType;
    constexpr static bool GetSide()
    {
        return ASK;
    }

    constexpr static AskType *GetType()
    {
        return (AskType *)nullptr;
    }

    constexpr static BidType *GetOtherType()
    {
        return (BidType *)nullptr;
    }

    constexpr static int GetSign()
    {
        return -1;
    }

    template <typename T>
    static T AddAway(T lhs, T rhs)
    {
        return lhs + rhs;
    }

    template <typename T>
    static T AddInner(T lhs, T rhs)
    {
        return lhs - rhs;
    }

    template <typename T>
    static bool IsInner(T lhs, T rhs)
    {
        return lhs < rhs;
    }

    template <typename T>
    static bool IsInnerOrEqual(T lhs, T rhs)
    {
        return lhs <= rhs;
    }

    template <typename T>
    static T IsCrossed(T lhs, T rhs)
    {
        return lhs >= rhs;
    }

    template <typename T>
    static T GetInner(T lhs, T rhs)
    {
        return std::min(lhs, rhs);
    }

    template <typename T>
    static T GetOuter(T lhs, T rhs)
    {
        return std::max(lhs, rhs);
    }
};


}  // namespace alphaone
#endif
