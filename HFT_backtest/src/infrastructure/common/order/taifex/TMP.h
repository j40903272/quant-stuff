#ifndef _TAIFEXTMP_PROTOCOL_H_
#define _TAIFEXTMP_PROTOCOL_H_

#include <arpa/inet.h>

#ifndef _TRADE_TAIFEXPROTOCOL_H_
#define _TRADE_TAIFEXPROTOCOL_H_

namespace alphaone
{

#ifdef __cpluscplus
extern "C"
{
#endif


#ifndef Ntohll
#define Ntohll(x) (((long)(ntohl((int)((x << 32) >> 32))) << 32) | ntohl(((int)(x >> 32))))
#endif
#ifndef Htonll
#define Htonll(x) Ntohll(x)
#endif

#define TMPSET_CHAR_N(DES, SRC, N) (memcpy((DES), (SRC), (N)))
#define TMPSET_UINT8(DES, SRC) ((DES) = (SRC))
#define TMPSET_INT8(DES, SRC) ((DES) = (SRC))
#define TMPSET_UINT16(DES, SRC) ((DES) = htons((SRC)))
#define TMPSET_INT16(DES, SRC) ((DES) = htons((SRC)))
#define TMPSET_UINT32(DES, SRC) ((DES) = htonl((SRC)))
#define TMPSET_INT32(DES, SRC) ((DES) = htonl((SRC)))
#define TMPSET_INT64(DES, SRC) ((DES) = Htonll((SRC)))
#define TMPSET_MSGLEN(DES, LEN)                                                                    \
    ((DES) = htons((LEN) - sizeof(unsigned short) - sizeof(unsigned char)))

#define TMPGET_MSGLEN(SRC)                                                                         \
    ((unsigned short)(ntohs(SRC)) + sizeof(unsigned short) + sizeof(unsigned char) -               \
     sizeof(TMPHdr_t))

#define TMPGET_INT8(SRC) ((char)(SRC))
#define TMPGET_UINT8(SRC) ((unsigned char)(SRC))
#define TMPGET_UINT16(SRC) ((unsigned short)ntohs((SRC)))
#define TMPGET_INT16(SRC) ((short)ntohs((SRC)))
#define TMPGET_UINT32(SRC) ((unsigned int)ntohl((SRC)))
#define TMPGET_INT32(SRC) ((int)ntohl((SRC)))
#define TMPGET_INT64(SRC) ((long)Ntohll((SRC)))

#pragma pack(1)
    /*
        R01=80, R02=133, R03=44, TPrice=12, R09=84
    */
    /* ====================================================== */
    /*
    char[n] n    char[n]
    uint8   1    unsigned char
    int8    1    char
    uint16  2    unsigned short
    int16   2    short
    uint32  4    unsigned int
    int32   4    int
    int64   8    long
    * */

    /* ====================================================== */
    /* status code / Error / Warning */
    /* warning :
     *           200 : time gap too large , over 0.5 sec (for R01/R04/R05 HBT)
     *           246 : time gap with taifex over 0.5 sec
     *           247 : time gap with taifex over 1   sec
     *           248 : the session network traffic over 80%
     *           249 : the session network traffic over 90%
     * error   :
     *
     * */
    /* ====================================================== */
    /* ====================================================== */
    /* tmp message flow              */
    /*  Client                            Server
     * ------------                      ---------------
     *     L10           ----->
     *                   <-----            L10
     *     L20           ----->
     *                   <-----            L30
     *     L40           ----->
     *                      <-----         L41
     *     L42              ----->
     *                   <-----            L50
     *     L60           ----->
     *
     *    Heartbeat
     *                   <-----            R04
     *     R05           ----->
     *
     *    order
     *     R01           ----->
     *                   <-----            R02
     *    market maker
     *     R09           ----->
     *                   <-----            L02
     *
     *
     *
     *
     * */
    /* ====================================================== */
    /* =====================================================================
     * enumerator
     * ===================================================================== */
    typedef enum _TMPMsgType
    {
        TMPMsgType_L10 = 10,
        TMPMsgType_L20 = 20,
        TMPMsgType_L30 = 30,
        TMPMsgType_L40 = 40,
        TMPMsgType_L41 = 41,
        TMPMsgType_L42 = 42,
        TMPMsgType_L50 = 50,
        TMPMsgType_L60 = 60,
        TMPMsgType_R01 = 101,
        TMPMsgType_R02 = 102,
        TMPMsgType_R03 = 103,
        TMPMsgType_R04 = 104,
        TMPMsgType_R05 = 105,
        TMPMsgType_R09 = 109,
        TMPMsgType_R11 = 111,
        TMPMsgType_R12 = 112
    } TMPMsgType;

    typedef enum _TMPTradeStatusType
    {
        TMPTradeStatusType_Unknow  = 0,
        TMPTradeStatusType_Stop    = 1,
        TMPTradeStatusType_Open    = 2,
        TMPTradeStatusType_Close   = 3,
        TMPTradeStatusType_PreOpen = 4
    } TMPTradeStatusType;

    /* =====================================================================
     * Data Define
     * ===================================================================== */
#define TMP_UDD_LEN 8
#define TMP_SYM_LEN 20
#define TMP_ORDNO_LEN 5

    /* SymbolFlag     */
#define TMP_SYM_NUM 1
#define TMP_SYM_TEXT 2

    /* ExecType       */
#define TMP_EXEC 'F'
#define TMP_EXEC_NEW '6'

    /* Order_source */
#define TMP_ORDSRC "D"

    /* Info_source */
#define TMP_INFOSRC "999"

    /* R11 broker->Taifex , order status query */
#define TMP_REQ_ORDSTATUS 0  /* query current status by group id. */
#define TMP_REG_ORDSTATUS 1  /* regist order status change event by group id. */
#define TMP_UREG_ORDSTATUS 2 /* unregist order status change event by group id. */
    /* =====================================================================
     * Tmp header
     * ===================================================================== */
    typedef struct _TMPMsgTime_t
    {
        int            epoch_s; /* second from Epoch(00:00:00 UTC 1970/1/1) */
        unsigned short ms;      /* millisecond */
    } TMPMsgTime_t;

    typedef struct _TMPHdr_t
    {
        unsigned short msg_length;
        unsigned int   MsgSeqNum;
        TMPMsgTime_t   msg_time;
        unsigned char  MessageType;
        unsigned short fcm_id;
        unsigned short session_id;
    } TMPHdr_t;

    typedef struct _TMPRTN_t
    {
        TMPHdr_t      hdr;
        unsigned char status_code;
        unsigned char CheckSum;
    } TMPRTN_t;

    /* =====================================================================
     * Tmp body  ( session level )
     * ===================================================================== */
    typedef struct _TMPL10_t
    {
        TMPHdr_t      hdr;
        unsigned char status_code;
        unsigned int  start_in_bound_num;
        unsigned char CheckSum;
    } TMPL10_t;

#define TMPL20_t TMPRTN_t
#define TMPL42_t TMPRTN_t
#define TMPL60_t TMPRTN_t

    typedef struct _TMPL30_t
    {
        TMPHdr_t       hdr;
        unsigned char  status_code;
        unsigned short append_no;
        unsigned int   end_out_bound_num;
        unsigned char  system_type;
        unsigned char  EncryptMethod;
        unsigned char  CheckSum;
    } TMPL30_t;

    typedef struct _TMPL40_t
    {
        TMPHdr_t       hdr;
        unsigned char  status_code;
        unsigned short append_no;
        unsigned short fcm_id;
        unsigned short session_id;
        unsigned char  system_type;
        unsigned char  ap_code;
        unsigned char  key_value;
        unsigned int   request_start_seq;
        unsigned char  cancel_order_sec;
        unsigned char  CheckSum;
    } TMPL40_t;

    typedef struct _TMPL41_t
    {
        TMPHdr_t      hdr;
        unsigned char status_code;
        unsigned char is_eof;
        unsigned int  file_size;
        unsigned char data; /* data length = hdr.msg_length - 15 - 6 = file_size + 1(checksum)
                                last char is checksum                    */
    } TMPL41_t;

    typedef struct _TMPL50_t
    {
        TMPHdr_t       hdr;
        unsigned char  status_code;
        unsigned char  HeartBtInt;
        unsigned short max_flow_ctrl_cnt;
        unsigned char  CheckSum;
    } TMPL50_t;

    /* =====================================================================
     * Tmp body  ( application level )
     * ===================================================================== */
#define TMPR04_t TMPRTN_t /* HBT (client -> server) */
#define TMPR05_t TMPRTN_t /* HBT (client <- server) */
                          /*
        unsigned short  msg_length;
        unsigned int    MsgSeqNum;
        TMPMsgTime_t    msg_time;
        unsigned char   MessageType;
        unsigned short  fcm_id;
        unsigned short  session_id;
*/
    /* Order Request R01 */
    typedef struct _TMPR01_t
    {
        TMPHdr_t       hdr;
        char           ExecType; /* 0:New, 4:Cxl, 5:Reduce Qty m:Modify Price */
        unsigned short cm_id;
        unsigned short fcm_id;
        char           order_no[TMP_ORDNO_LEN];
        unsigned int   ord_id;
        char           user_define[TMP_UDD_LEN];
        unsigned char  symbol_type; /* 2:text */
        char           sym[TMP_SYM_LEN];
        int            Price;
        unsigned short qty;
        unsigned int   investor_acno;
        char           investor_flag;
        unsigned char  Side;           /* 1:buy, 2:sell */
        unsigned char  OrdType;        /* 1:market, 2:limit */
        unsigned char  TimeInForce;    /* 4:FOK, 3:IOC 0:ROD */
        char           PositionEffect; /* O:open, C:close, D:daytrade */
        char           order_source[1];
        char           info_source[3];
        unsigned char  CheckSum;
    } TMPR01_t;

    /* Order Request  R09 */
    typedef struct _TMPR09_t
    {
        TMPHdr_t       hdr;
        char           ExecType; /* 0:New, 4:Cxl, 5:Reduce Qty */
        unsigned short cm_id;
        unsigned short fcm_id;
        char           order_no[TMP_ORDNO_LEN];
        unsigned int   ord_id;
        char           user_define[TMP_UDD_LEN];
        unsigned char  symbol_type; /* 2:text */
        char           sym[TMP_SYM_LEN];
        int            BidPx;
        int            OfferPx;
        unsigned short BidSize;
        unsigned short OfferSize;
        unsigned int   investor_acno;
        char           investor_flag;
        unsigned char  TimeInForce;    /* 0:ROD 8:auto-cxl */
        char           PositionEffect; /* 9:market maker */
        char           order_source[1];
        char           info_source[3];
        unsigned char  CheckSum;
    } TMPR09_t;

    /* Order Report */
    typedef struct _TMPR02_t
    {
        TMPHdr_t       hdr;
        unsigned char  status_code;
        char           ExecType; /* 0:New, 4:Cxl, 5:Reduce Qty m:Modify Price */
        unsigned short cm_id;
        unsigned short fcm_id;
        char           order_no[TMP_ORDNO_LEN];
        unsigned int   ord_id;
        char           user_define[TMP_UDD_LEN];
        unsigned char  symbol_type; /* 2:text */
        char           sym[TMP_SYM_LEN];
        int            Price;
        unsigned short qty;
        unsigned int   investor_acno;
        char           investor_flag;
        unsigned char  Side;           /* 1:buy, 2:sell */
        unsigned char  OrdType;        /* 1:market, 2:limit */
        unsigned char  TimeInForce;    /* 4:FOK, 3:IOC 0:ROD 8:Quote */
        char           PositionEffect; /* 9:Market Maker */
        int            LastPx;
        unsigned short LastQty;
        long           px_subtotal;
        unsigned short CumQty;
        unsigned short LeavesQty;
        unsigned short before_qty;
        unsigned char  leg_side_1;
        unsigned char  leg_side_2;
        int            leg_px_1;
        int            leg_px_2;
        unsigned short leg_qty_1;
        unsigned short leg_qty_2;
        TMPMsgTime_t   org_trans_time;
        TMPMsgTime_t   TransactTime;
        unsigned char  target_id;
        unsigned int   uniq_id;
        unsigned int   rpt_seq;
        unsigned char  protocol_type;
        unsigned char  CheckSum;
    } TMPR02_t;

    /* Error Report */
    typedef struct _TMPR03_t
    {
        TMPHdr_t       hdr;
        unsigned char  status_code;
        char           ExecType; /* 0:New, 4:Cxl, 5:Reduce Qty m:Modify Price */
        unsigned short fcm_id;
        char           order_no[TMP_ORDNO_LEN];
        unsigned int   ord_id;
        char           user_define[TMP_UDD_LEN];
        unsigned int   rpt_seq;
        unsigned char  Side; /* 1:buy, 2:sell */
        unsigned char  CheckSum;
    } TMPR03_t;

    /* Taifex Status Query */
    typedef struct _TMPR11_t
    {
        TMPHdr_t       hdr;
        unsigned char  status_code;
        unsigned short TradeReqID;
        unsigned char  flow_group_no;
        unsigned char  SubscriptionRequestType;
        unsigned char  CheckSum;
    } TMPR11_t;

    typedef struct _TMPR12_t
    {
        TMPHdr_t       hdr;
        unsigned char  status_code;
        unsigned short TradeReqID;
        unsigned char  flow_group_no;
        unsigned char  TradeStatus;
        unsigned char  CheckSum;
    } TMPR12_t;

    typedef union _TMPMsg_t
    {
        TMPHdr_t Hdr;
        TMPL10_t L10;
        TMPL20_t L20;
        TMPL30_t L30;
        TMPL40_t L40;
        TMPL41_t L41;
        TMPL42_t L42;
        TMPL50_t L50;
        TMPL60_t L60;
        TMPR01_t R01;
        TMPR02_t R02;
        TMPR03_t R03;
        TMPR04_t R04;
        TMPR05_t R05;
        TMPR09_t R09;
        TMPR11_t R11;
        TMPR12_t R12;
        char     RcvBuffer[65535];
    } TMPMsg_t;

#pragma pack()
#ifdef __cpluscplus
}
#endif

void     TMPSetCheckSum(unsigned char *Checksum, const void *Data, size_t DataLen);
TMPL10_t MakeL10(unsigned short fcm_id, unsigned short session_id);
TMPL20_t MakeL20(unsigned short fcm_id, unsigned short session_id);
TMPL40_t MakeL40(unsigned short fcm_id, unsigned short session_id, unsigned char UChar,
                 unsigned int AppendNo, int end_out_bound_num, int password);
TMPL42_t MakeL42(unsigned short fcm_id, unsigned short session_id);
TMPL60_t MakeL60(unsigned short fcm_id, unsigned short session_id);
TMPR04_t MakeR04(unsigned short fcm_id, unsigned short session_id);
TMPR05_t MakeR05(unsigned short fcm_id, unsigned short session_id);
TMPR11_t MakeR11(unsigned short fcm_id, unsigned short session_id, unsigned short TradeReqID,
                 unsigned char flow_group_no, unsigned char SubscriptionRequestType);
void     TMPHdrSet(TMPHdr_t *Hdr, unsigned char MsgType, size_t MsgLen);

#endif /* _TRADE_TAIFEXPROTOCOL_H_ */

#pragma pack(1)

/* Taifex File */
typedef struct _TMPP06_t
{
    char fcm_no[7];
    char fcm_no_id[5];
} TMPP06_t;

typedef struct _TMPP07_t
{
    char connection_svr_name_And_port_no[200 + 1 + 5];
    char linefeed[1];
} TMPP07_t;

typedef struct _TMPP12_t
{
    char fcm_no[7];
    char connection_type[1];
    char session_id[5];
    char fcm_ip[15];
    char fcm_socket_port[5];
    char filler[47];
} TMPP12_t;

struct P08
{
    char prod_id_s[10];
    char settle_date[6];
    char strike_price[9];
    char call_put_code;
    char begin_date[8];
    char end_date[8];
    char raise_price1[9];
    char fall_price1[9];
    char premium[9];
    char raise_price2[9];
    char fall_price2[9];
    char raise_price3[9];
    char fall_price3[9];
    char prod_kind;
    char accept_quote_flag;
    char decimal_locator;
    char pseq[5];
    char flow_group[2];
    char delivery_date[8];
    char strike_price_decimal_locator;
    char filler[36];
};
#pragma pack()
}
#endif