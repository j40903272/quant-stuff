#ifndef _TWSE_FIXFORMAT_H_
#define _TWSE_FIXFORMAT_H_

#define HEADER_FIXED_SIZE 16
#define TRAILER_FIXED_SIZE 7
namespace alphaone
{
#pragma pack(1)
struct TWSEFixHeader
{
    // tag + '=' + data size + split
    char BeginString[1 + 1 + 7 + 1];
    char BodyLength[1 + 1 + 3 + 1];
    char MsgType[2 + 1 + 1 + 1];
    char MsgSeqNum[2 + 1 + 8 + 1];
    char SenderCompID[2 + 1 + 7 + 1];
    char SenderSubID[2 + 1 + 4 + 1];
    char SendingTime[2 + 1 + 21 + 1];
    char TargetCompID[2 + 1 + 4 + 1];
    char TargetSubID[2 + 1 + 1 + 1];
};

struct TWSEFixTrailer
{
    char CheckSum[2 + 1 + 3 + 1];
};

struct TWSEFixLogon
{
    TWSEFixHeader  Header;
    char           EncryptMethod[2 + 1 + 1 + 1];
    char           HeartBtInt[3 + 1 + 2 + 1];
    char           RawDataLength[2 + 1 + 1 + 1];
    char           RawData[2 + 1 + 5 + 1];
    TWSEFixTrailer Trailer;
};

struct TWSEFixHeartbeat
{
    TWSEFixHeader  Header;
    char           TestReqID[3 + 1 + 27 + 1];
    TWSEFixTrailer Trailer;
};

struct TWSEFixTestRequest
{
    TWSEFixHeader  Header;
    char           TestReqID[3 + 1 + 9 + 1];
    TWSEFixTrailer Trailer;
};

struct TWSEFixReSendRequest
{
    TWSEFixHeader  Header;
    char           BeginSeqNo[1 + 1 + 10 + 1];
    char           EndSeqNo[2 + 1 + 10 + 1];
    TWSEFixTrailer Trailer;
};

struct TWSEFixNewOrder
{
    TWSEFixHeader  Header;
    char           ClOrdID[2 + 1 + 12 + 1];  // 7798J1000001
    char           OrderID[2 + 1 + 5 + 1];
    char           Account[1 + 1 + 7 + 1];
    char           Symbol[2 + 1 + 6 + 1];
    char           Side[2 + 1 + 1 + 1];
    char           TransactTime[2 + 1 + 21 + 1];
    char           OrderQty[2 + 1 + 3 + 1];
    char           OrdType[2 + 1 + 1 + 1];
    char           TimeInForce[2 + 1 + 1 + 1];
    char           Price[2 + 1 + 10 + 1];
    char           TwseIvacnoFlag[5 + 1 + 1 + 1];
    char           TwseOrdType[5 + 1 + 1 + 1];
    char           TwseExCode[5 + 1 + 1 + 1];
    char           TwseRejStaleOrd[5 + 1 + 1 + 1];
    TWSEFixTrailer Trailer;
};

struct TWSEFixCancelOrder
{
    TWSEFixHeader  Header;
    char           OrigClOrdID[2 + 1 + 12 + 1];
    char           ClOrdID[2 + 1 + 12 + 1];
    char           OrderID[2 + 1 + 5 + 1];
    char           Account[1 + 1 + 7 + 1];
    char           Symbol[2 + 1 + 6 + 1];
    char           Side[2 + 1 + 1 + 1];
    char           TransactTime[2 + 1 + 21 + 1];
    char           TwseIvacnoFlag[5 + 1 + 1 + 1];
    char           TwseExCode[5 + 1 + 1 + 1];
    char           TwseRejStaleOrd[5 + 1 + 1 + 1];
    TWSEFixTrailer Trailer;
};

struct TWSEFixModifyOrder
{
    TWSEFixHeader  Header;
    char           OrigClOrdID[2 + 1 + 12 + 1];
    char           ClOrdID[2 + 1 + 12 + 1];
    char           OrderID[2 + 1 + 5 + 1];
    char           Account[1 + 1 + 7 + 1];
    char           Symbol[2 + 1 + 6 + 1];
    char           Side[2 + 1 + 1 + 1];
    char           TransactTime[2 + 1 + 21 + 1];
    char           OrderQty[2 + 1 + 3 + 1];
    char           OrdType[2 + 1 + 1 + 1];
    char           Price[2 + 1 + 10 + 1];
    char           TwseIvacnoFlag[5 + 1 + 1 + 1];
    char           TwseExCode[5 + 1 + 1 + 1];
    char           TwseRejStaleOrd[5 + 1 + 1 + 1];
    TWSEFixTrailer Trailer;
};

struct TWSEFixOrder
{
    int                Type;
    TWSEFixNewOrder    New;
    TWSEFixCancelOrder Cancel;
    TWSEFixModifyOrder Modify;
};

#pragma pack()
}  // namespace alphaone
#endif