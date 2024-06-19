#ifndef BRANCH_H
#define BRANCH_H

#ifdef __GNUC__

#define BRANCH_LIKELY(x) __builtin_expect(static_cast<bool>(x), true)
#define BRANCH_UNLIKELY(x) __builtin_expect(static_cast<bool>(x), false)
#define VERY_LIKELY(x) __builtin_expect_with_probability(x, 1, 0.99)
#define VERY_UNLIKELY(x) __builtin_expect_with_probability(x, 0, 0.99)

#else

#define BRANCH_LIKELY(x) (x)
#define BRANCH_UNLIKELY(x) (x)
#define VERY_LIKELY(x) (x)
#define VERY_UNLIKELY(x) (x)

#endif

#ifdef __clang__

#define UNPREDICTABLE(x) __builtin_unpredictable(x)

#else

#define UNPREDICTABLE(x) (x)

#endif

#endif
