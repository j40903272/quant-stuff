#ifndef _PACKET_LOG_STRUCT_H
#define _PACKET_LOG_STRUCT_H

#include "infrastructure/common/twone/ringbuffer/RingBuffer.h"

enum class PacketLogType : int
{
    INVALID = 0,
    TAIFEX  = 1,
    TWSE    = 2,
    TPRICE  = 99,
};


enum class PacketLogChannelID : int
{
    INVALID = 0,
    FUTURE  = 1,
    OPTION  = 2,
    TSE     = 3,
    OTC     = 4
};

enum class PacketLogExecType : char
{
    INVALID = ' ',
    NEW     = '0',
    CANCEL  = '4',
};

struct PacketLogStruct
{
    int      Type;  // 1=Taifex, 99=T
    int      SeqNum;
    int      LoopCount;
    int      ChannelID;
    char     ChannelSeq[5];
    char *   OrderNo;
    char     OrderNoStr[5];
    char     UserDefine[8];
    uint32_t UniqueID;
    char     ExecType;
    int      OrderTo;

    twone::RingBuffer *RingBuffer_PacketLog;
};
#endif