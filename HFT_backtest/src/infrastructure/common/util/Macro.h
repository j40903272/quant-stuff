#ifndef _MACRO_H_
#define _MACRO_H_

namespace alphaone
{

#define FOR_EACH_0(MACRO)
#define FOR_EACH_1(MACRO, X) MACRO(X)
#define FOR_EACH_2(MACRO, X, ...) MACRO(X) FOR_EACH_1(MACRO, __VA_ARGS__)
#define FOR_EACH_3(MACRO, X, ...) MACRO(X) FOR_EACH_2(MACRO, __VA_ARGS__)
#define FOR_EACH_4(MACRO, X, ...) MACRO(X) FOR_EACH_3(MACRO, __VA_ARGS__)
#define FOR_EACH_5(MACRO, X, ...) MACRO(X) FOR_EACH_4(MACRO, __VA_ARGS__)
#define FOR_EACH_6(MACRO, X, ...) MACRO(X) FOR_EACH_5(MACRO, __VA_ARGS__)
#define FOR_EACH_7(MACRO, X, ...) MACRO(X) FOR_EACH_6(MACRO, __VA_ARGS__)
#define FOR_EACH_8(MACRO, X, ...) MACRO(X) FOR_EACH_7(MACRO, __VA_ARGS__)
#define FOR_EACH_9(MACRO, X, ...) MACRO(X) FOR_EACH_8(MACRO, __VA_ARGS__)
#define FOR_EACH_10(MACRO, X, ...) MACRO(X) FOR_EACH_9(MACRO, __VA_ARGS__)
#define FOR_EACH_11(MACRO, X, ...) MACRO(X) FOR_EACH_10(MACRO, __VA_ARGS__)
#define FOR_EACH_12(MACRO, X, ...) MACRO(X) FOR_EACH_11(MACRO, __VA_ARGS__)
#define FOR_EACH_13(MACRO, X, ...) MACRO(X) FOR_EACH_12(MACRO, __VA_ARGS__)
#define FOR_EACH_14(MACRO, X, ...) MACRO(X) FOR_EACH_13(MACRO, __VA_ARGS__)

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, NAME, ...) NAME
#define FOR_EACH(ACT, ...)                                                                         \
    GET_MACRO(__VA_ARGS__, FOR_EACH_14, FOR_EACH_13, FOR_EACH_12, FOR_EACH_11, FOR_EACH_10,        \
              FOR_EACH_9, FOR_EACH_8, FOR_EACH_7, FOR_EACH_6, FOR_EACH_5, FOR_EACH_4, FOR_EACH_3,  \
              FOR_EACH_2)                                                                          \
    (ACT, __VA_ARGS__)

#define DISALLOW_COPY_AND_ASSIGN(ClassName)                                                        \
    ClassName(const ClassName &);                                                                  \
    void operator=(const ClassName &);

}  // namespace alphaone


#endif