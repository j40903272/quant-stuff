#include <sys/types.h>
#ifndef _UNPACKBCD_H_
#define _UNPACKBCD_H_
inline int Decode1(char *data)
{
    return ((data[0]) & 0x0f) + ((data[0] >> 4) & 0x0f) * 10;
}

inline int Decode2(char *data)
{
    return ((data[1]) & 0x0f) + ((data[1] >> 4) & 0x0f) * 10 + ((data[0]) & 0x0f) * 100 +
           ((data[0] >> 4) & 0x0f) * 1000;
}

inline int Decode3(char *data)
{
    return ((data[2]) & 0x0f) + ((data[2] >> 4) & 0x0f) * 10 + ((data[1]) & 0x0f) * 100 +
           ((data[1] >> 4) & 0x0f) * 1000 + ((data[0]) & 0x0f) * 10000 +
           ((data[0] >> 4) & 0x0f) * 100000;
}

inline int Decode4(char *data)
{
    return ((data[3]) & 0x0f) + ((data[3] >> 4) & 0x0f) * 10 + ((data[2]) & 0x0f) * 100 +
           ((data[2] >> 4) & 0x0f) * 1000 + ((data[1]) & 0x0f) * 10000 +
           ((data[1] >> 4) & 0x0f) * 100000 + ((data[0]) & 0x0f) * 1000000 +
           ((data[0] >> 4) & 0x0f) * 10000000;
}

inline int Decode5(char *data)
{
    return ((data[4]) & 0x0f) + ((data[4] >> 4) & 0x0f) * 10 + ((data[3]) & 0x0f) * 100 +
           ((data[3] >> 4) & 0x0f) * 1000 + ((data[2]) & 0x0f) * 10000 +
           ((data[2] >> 4) & 0x0f) * 100000 + ((data[1]) & 0x0f) * 1000000 +
           ((data[1] >> 4) & 0x0f) * 10000000 + ((data[0]) & 0x0f) * 100000000 +
           ((data[0] >> 4) & 0x0f) * 1000000000;
}

inline int64_t Decode6(char *data)
{
    return ((data[5]) & 0x0f) + ((data[5] >> 4) & 0x0f) * 10 + ((data[4]) & 0x0f) * 100 +
           ((data[4] >> 4) & 0x0f) * 1000 + ((data[3]) & 0x0f) * 10000 +
           ((data[3] >> 4) & 0x0f) * 100000 + ((data[2]) & 0x0f) * 1000000 +
           ((data[2] >> 4) & 0x0f) * 10000000 + ((int64_t)((data[1]) & 0x0f)) * (int64_t)100000000 +
           ((int64_t)((data[1] >> 4) & 0x0f)) * (int64_t)1000000000 +
           ((int64_t)((data[0]) & 0x0f)) * (int64_t)10000000000 +
           ((int64_t)((data[0] >> 4) & 0x0f)) * (int64_t)100000000000;
}
#endif