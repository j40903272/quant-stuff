#ifndef _TWSE_H
#define _TWSE_H
namespace alphaone
{
typedef enum _enMktLmtFlag
{
    enDelayMatchDown = 1 << 0,
    enDelayMatchUp   = 1 << 1,
    enMatchDown      = 1 << 6,
    enMatchUp        = 1 << 7
} enMktLmtFlg;

#pragma pack(1)
typedef struct _TWSEDataHdr_t
{
    char EscCode[1]; /* ascii 27 */
    char BodyLength[2];
    char BusinessKind[1];
    char Format[1];
    char VersionNo[1];
    char InformationSequenceNo[4];
} TWSEDataHdr_t;
#pragma pack()

#pragma pack(1)
typedef struct _TWSERecord_RealTIme_t
{
    char Price[5];
    char Qty[4];
} TWSERecord_RealTime_t;
#pragma pack()

#pragma pack(1)
typedef struct _TWSEDataFormat6_RealTime_t
{
    TWSEDataHdr_t         Hdr;
    char                  ProductID[6];
    char                  MatchTime[6];
    char                  InformationPrompt[1];
    char                  MarketLimitPrompt[1];
    char                  StatusPrompt[1];
    char                  TotalMatchQty[4];
    TWSERecord_RealTime_t Records[11];

} TWSEDataFormat6_RealTime_t;
#pragma pack()


#define TWSEDataHdrLen sizeof(TWSEDataHdr_t)
#define TWSERecordLen sizeof(TWSERecord_t)
#define TWSEDataFormat6Len sizeof(TWSEDataFormat6_t)
#define Format6IsSimulated(Format6) ((unsigned char)((Format6)->StatusPrompt[0]) & 128)

#define TWSE_SHOW_MATCHPRICETMASK 128  // 10000000
#define TWSE_SHOW_BUYCNTMASK 112       // 01110000
#define TWSE_SHOW_SELLCNTMASK 14       // 00001110
}  // namespace alphaone
#endif