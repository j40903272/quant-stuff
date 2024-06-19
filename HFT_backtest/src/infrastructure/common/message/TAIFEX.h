#ifndef _MARKETDATA_TAIFEX_H_
#define _MARKETDATA_TAIFEX_H_
namespace alphaone
{
#pragma pack(1)
typedef struct _TXMarketDataHdr_t
{
    char EscCode[1];
    char TransmissionCode[1];
    char MessageKind[1];
    char InformationTime[6];
    char InformationSequenceNo[4];
    char VersionNo[1];
    char BodyLength[2];
} TXMarketDataHdr_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataRecord_t
{
    char Sign[1];
    char Price[5];
    char Qty[4];
} TXMarketDataRecord_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataRepeatingRecord_t
{
    char Sign[1];
    char Price[5];
    char Qty[2];
} TXMarketDataRepeatingRecord_t;
#pragma pack()

#pragma pack(1)
// len(I020)=625
typedef struct _TXMarketDataI020_t
{
    TXMarketDataHdr_t             Hdr;
    char                          ProductID[20];
    char                          MatchTime[6];
    TXMarketDataRecord_t          FirstMatchInfo;
    char                          FurtherMatchItemNo[1];
    TXMarketDataRepeatingRecord_t FurtherMatchInfo[72];
} TXMarketDataI020_t;
#pragma pack()

#pragma pack(1)
// len(I080=134)
typedef struct _TXMarketDataI080_t
{
    TXMarketDataHdr_t    Hdr;
    char                 ProductID[20];
    TXMarketDataRecord_t BuyOrder1;
    TXMarketDataRecord_t BuyOrder2;
    TXMarketDataRecord_t BuyOrder3;
    TXMarketDataRecord_t BuyOrder4;
    TXMarketDataRecord_t BuyOrder5;
    TXMarketDataRecord_t SellOrder1;
    TXMarketDataRecord_t SellOrder2;
    TXMarketDataRecord_t SellOrder3;
    TXMarketDataRecord_t SellOrder4;
    TXMarketDataRecord_t SellOrder5;
} TXMarketDataI080_t;

typedef struct _TXMarketDataHdr_RealTime_t
{
    char EscCode[1];
    char TransmissionCode[1];
    char MessageKind[1];
    char InformationTime[6];
    char ChannelID[2];
    char ChannelSeq[5];
    char VersionNo[1];
    char BodyLength[2];
} TXMarketDataHdr_RealTime_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataRecord_RealTime_t
{
    char Sign[1];
    char Price[5];
    char Qty[4];
} TXMarketDataRecord_RealTime_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataRepeatingRecord_RealTime_t
{
    char Sign[1];
    char Price[5];
    char Qty[2];
} TXMarketDataRepeatingRecord_RealTime_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataI024_t
{
    TXMarketDataHdr_RealTime_t             Hdr;
    char                                   ProductID[20];
    char                                   ProductMsgSeq[5];
    char                                   CalculatedFlag[1];
    char                                   MatchTime[6];
    TXMarketDataRecord_RealTime_t          FirstMatchInfo;
    char                                   FurtherMatchItemNo[1];
    TXMarketDataRepeatingRecord_RealTime_t FurtherMatchInfo[72];
} TXMarketDataI024_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataI025_t
{
    TXMarketDataHdr_RealTime_t Hdr;
    char                       ProductID[20];
    char                       ProductMsgSeq[5];
} TXMarketDataI025_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataI081RepeatingRecord_RealTime_t
{
    char MD_UPDATE_ACTION[1];
    char MD_ENTRY_TYPE[1];
    char Sign[1];
    char Price[5];
    char Qty[4];
    char PriceLevel[1];
} TXMarketDataI081RepeatingRecord_RealTime_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataI083RepeatingRecord_RealTime_t
{
    char MD_ENTRY_TYPE[1];
    char Sign[1];
    char Price[5];
    char Qty[4];
    char PriceLevel[1];
} TXMarketDataI083RepeatingRecord_RealTime_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataI081_t
{
    TXMarketDataHdr_RealTime_t                 Hdr;
    char                                       ProductID[20];
    char                                       ProductMsgSeq[5];
    char                                       NO_MD_ENTRIES[1];
    TXMarketDataI081RepeatingRecord_RealTime_t FurtherInfo[255];
} TXMarketDataI081_t;
#pragma pack()

#pragma pack(1)
typedef struct _TXMarketDataI083_t
{
    TXMarketDataHdr_RealTime_t                 Hdr;
    char                                       ProductID[20];
    char                                       ProductMsgSeq[5];
    char                                       Calculated_Flag[1];
    char                                       NO_MD_ENTRIES[1];
    TXMarketDataI083RepeatingRecord_RealTime_t FurtherInfo[255];
} TXMarketDataI083_t;

#pragma pack()

typedef struct _TXMarketDataI084_t
{
    TXMarketDataHdr_RealTime_t Hdr;
    char                       MsgType[1];
} TXMarketDataI084_t;

typedef struct _TXMarketDataI084_A_t
{
    char Last_Seq[5];
} TXMarketDataI084_A_t;

typedef struct _TXMarketDataI084_Z_t
{
    char Last_Seq[5];
} TXMarketDataI084_Z_t;

typedef struct _TXMarketDataI084_O_Header_t
{
    char No_Entries[1];
} TXMarketDataI084_O_Header_t;

typedef struct _TXMarketDataI084_O_Product_t
{
    char ProductID[20];
    char Last_Prod_Msg_Seq[5];
    char NO_MD_ENTRIES[1];
} TXMarketDataI084_O_Product_t;

typedef struct _TXMarketDataI084_O_Info_t
{
    char MD_ENTRY_TYPE[1];
    char Sign[1];
    char Price[5];
    char Qty[4];
    char PriceLevel[1];
} TXMarketDataI084_O_Info_t;

#pragma pack()
}  // namespace alphaone
#endif