#include "TaifexOrderSession.h"

#include "infrastructure/common/order/taifex/TMP.h"
#include "infrastructure/common/symbol/Symbol.h"

#include <string.h>

namespace alphaone
{
TaifexOrderSession::TaifexOrderSession()
{
}

TaifexOrderSession::~TaifexOrderSession()
{
}

void TaifexOrderSession::Init(int session_id, TAIFEX_ORDERSESSION_TYPE type)
{
    session_id_ = session_id;
    type_       = type;

    r01_ = twone::RingBuffer((int)twone::TAIFEX_ORDER_INDEX_BASEINDEX::R01 + session_id,
                             (int)twone::TAIFEX_ORDER_BLOCK_BASEINDEX::R01 + session_id, 128, 2048,
                             -1, 0);

    r09_ = twone::RingBuffer((int)twone::TAIFEX_ORDER_INDEX_BASEINDEX::R09 + session_id,
                             (int)twone::TAIFEX_ORDER_BLOCK_BASEINDEX::R09 + session_id, 128, 2048,
                             -1, 0);

    r02_ = twone::RingBuffer((int)twone::TAIFEX_ORDER_INDEX_BASEINDEX::R02 + session_id,
                             (int)twone::TAIFEX_ORDER_BLOCK_BASEINDEX::R02 + session_id, 192, 2048,
                             -1, 0);
    r03_ = twone::RingBuffer((int)twone::TAIFEX_ORDER_INDEX_BASEINDEX::R03 + session_id,
                             (int)twone::TAIFEX_ORDER_BLOCK_BASEINDEX::R03 + session_id, 64, 64, -1,
                             0);
}

void TaifexOrderSession::NewOrder(const char *pid, int price, int qty, TAIFEX_ORDER_SIDE side,
                                  TAIFEX_ORDER_TIMEINFORCE    timeInForce,
                                  TAIFEX_ORDER_POSITIONEFFECT positionEffect, int subSessionIndex,
                                  TaifexOrderStatus *order_status, unsigned int account,
                                  char accountFlag, const char *pUserDefine)
{


    TMPR01_t *packet = (TMPR01_t *)r01_.GetNextAddress();
    TMPSET_UINT8(packet->ExecType, '0');
    TMPSET_CHAR_N(&packet->order_no[0], &order_status->OrderNo[0], 5);
    TMPSET_CHAR_N(&packet->sym[0], pid, SYMBOLID_LENGTH);
    TMPSET_CHAR_N(&packet->PositionEffect, &positionEffect, 1);
    TMPSET_UINT32(packet->Price, price);
    TMPSET_UINT8(packet->Side, (unsigned char)side);
    TMPSET_UINT16(packet->qty, qty);
    TMPSET_UINT8(packet->TimeInForce, (unsigned char)timeInForce);
    TMPSET_CHAR_N(&packet->user_define[0], pUserDefine, 8);
    TMPSET_UINT16(packet->hdr.msg_time.ms, subSessionIndex);

    TMPSET_UINT32(packet->investor_acno, account);
    TMPSET_UINT8(packet->investor_flag, accountFlag);

    r01_.AddReadyIndex(1);
    r01_.AddNextIndex(1);

    return;
}


void TaifexOrderSession::CancelOrder(const char *orderno, const char *pid, TAIFEX_ORDER_SIDE side,
                                     int subSessionIndex, const char *pUserDefine)
{
    TMPR01_t *packet = (TMPR01_t *)r01_.GetNextAddress();

    TMPSET_UINT8(packet->ExecType, '4');
    TMPSET_CHAR_N(&packet->order_no[0], &orderno[0], 5);
    TMPSET_CHAR_N(&packet->sym[0], pid, SYMBOLID_LENGTH);
    TMPSET_UINT8(packet->Side, (unsigned char)side);
    TMPSET_UINT8(packet->TimeInForce, 0);
    TMPSET_CHAR_N(&packet->user_define[0], pUserDefine, 8);
    TMPSET_UINT16(packet->hdr.msg_time.ms, subSessionIndex);

    r01_.AddReadyIndex(1);
    r01_.AddNextIndex(1);
}


void TaifexOrderSession::NewDoubleOrder(const char *pid, int bidprice, int bidqty, int askprice,
                                        int askqty, TAIFEX_ORDER_TIMEINFORCE timeInForce,
                                        TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                                        int subSessionIndex, TaifexOrderStatus *order_status,
                                        unsigned int account, char accountFlag,
                                        const char *pUserDefine)
{
    TMPR09_t *packet = (TMPR09_t *)r09_.GetNextAddress();
    TMPSET_UINT8(packet->ExecType, '0');
    TMPSET_CHAR_N(&packet->order_no[0], &order_status->OrderNo[0], 5);
    TMPSET_CHAR_N(&packet->sym[0], pid, SYMBOLID_LENGTH);
    TMPSET_CHAR_N(&packet->PositionEffect, &positionEffect, 1);

    TMPSET_UINT32(packet->BidPx, bidprice);
    TMPSET_UINT16(packet->BidSize, bidqty);

    TMPSET_UINT32(packet->OfferPx, askprice);
    TMPSET_UINT16(packet->OfferSize, askqty);

    TMPSET_UINT8(packet->TimeInForce, (unsigned char)timeInForce);
    TMPSET_CHAR_N(&packet->user_define[0], pUserDefine, 8);
    TMPSET_UINT16(packet->hdr.msg_time.ms, subSessionIndex);

    TMPSET_UINT32(packet->investor_acno, account);
    TMPSET_UINT8(packet->investor_flag, accountFlag);

    r09_.AddReadyIndex(1);
    r09_.AddNextIndex(1);
}

void TaifexOrderSession::ModifyDoubleOrder(const char *orderno, const char *pid, int bidprice,
                                           int askprice, int sessionIndex, int subSessionIndex,
                                           const char *pUserDefine)
{
    TMPR09_t *packet = (TMPR09_t *)r09_.GetNextAddress();
    TMPSET_UINT8(packet->ExecType, 'M');
    TMPSET_CHAR_N(&packet->order_no[0], orderno, 5);
    TMPSET_CHAR_N(&packet->sym[0], pid, SYMBOLID_LENGTH);

    TMPSET_UINT32(packet->BidPx, bidprice);

    TMPSET_UINT32(packet->OfferPx, askprice);

    TMPSET_CHAR_N(&packet->user_define[0], pUserDefine, 8);
    TMPSET_UINT16(packet->hdr.msg_time.ms, subSessionIndex);

    r09_.AddReadyIndex(1);
    r09_.AddNextIndex(1);
}

void TaifexOrderSession::CancelDoubleOrder(const char *orderno, const char *pid,
                                           int subSessionIndex, const char *pUserDefine)
{
    TMPR09_t *packet = (TMPR09_t *)r09_.GetNextAddress();

    TMPSET_UINT8(packet->ExecType, '4');
    TMPSET_CHAR_N(&packet->order_no[0], &orderno[0], 5);
    TMPSET_CHAR_N(&packet->sym[0], pid, SYMBOLID_LENGTH);
    TMPSET_UINT8(packet->TimeInForce, 0);
    TMPSET_CHAR_N(&packet->user_define[0], pUserDefine, 8);
    TMPSET_UINT16(packet->hdr.msg_time.ms, subSessionIndex);

    r09_.AddReadyIndex(1);
    r09_.AddNextIndex(1);
}

int TaifexOrderSession::GetOrderStatusIndex(std::vector<TaifexOrderStatus> &orderstatus_list)
{
    int ret = 0;

    char *pTmp  = NULL;
    char *pTmp2 = NULL;

    char filename[255];

    sprintf(filename, "/home/lgt/AP/twone/OrderExecutorEx/LastOrderNo/%d", GetSessionID());

    printf("filename=%s\n", filename);
    FILE *fp             = fopen(filename, "r");
    char  fileorderno[6] = {0};
    memset(fileorderno, 0, 6);
    if (fp != NULL)
    {
        // fscanf(fp, "%s", &fileorderno[0]);
        fclose(fp);
    }

    if (r01_.GetReadyIndex() >= 0)
    {
        int index = 0;
        while (1)
        {
            int realIndex = r01_.GetReadyIndex() - index;
            if (realIndex < 0)
            {
                break;
            }
            TMPR01_t *r01 = (TMPR01_t *)r01_.GetAddress(realIndex);

            if (r01->ExecType == '0')
            {
                pTmp = r01->order_no;
                break;
            }
            index++;
        }
    }

    if (r09_.GetReadyIndex() >= 0)
    {
        int index = 0;
        while (1)
        {
            int realIndex = r09_.GetReadyIndex() - index;
            if (realIndex < 0)
            {
                break;
            }
            TMPR09_t *r09 = (TMPR09_t *)r09_.GetAddress(realIndex);

            if (r09->ExecType == '0')
            {
                if (pTmp != NULL)
                {
                    int r01Int = OrderNoToInt(pTmp);
                    int r09Int = OrderNoToInt(r09->order_no);
                    if (r09Int > r01Int)
                    {
                        pTmp = r09->order_no;
                        break;
                    }
                }
                else
                {
                    pTmp = r09->order_no;
                    break;
                }
            }
            index++;
        }
    }

    int ordno1 = (pTmp == NULL) ? 0 : OrderNoToInt(pTmp);

    int ordno2 = OrderNoToInt(fileorderno);
    pTmp2      = pTmp;

    printf("FileOrderNo=%c%c%c%c%c\n", fileorderno[0], fileorderno[1], fileorderno[2],
           fileorderno[3], fileorderno[4]);

    if (pTmp != NULL)
    {
        printf("MemoryOrderNo=%c%c%c%c%c\n", pTmp[0], pTmp[1], pTmp[2], pTmp[3], pTmp[4]);
    }

    if (ordno2 > ordno1)
    {
        pTmp = &fileorderno[0];
    }

    if (pTmp != NULL)
    {
        for (unsigned int i = 0; i < orderstatus_list.size(); ++i)
        {
            if (memcmp(&pTmp[0], &orderstatus_list[i].OrderNo[0], 5) == 0)
            {
                ret = i + 1;
                break;
            }
        }
    }

    //如果檔案的OrderNo比記憶體的大而且在設定內找不到，強制使用記憶體的資料
    if (ordno2 > ordno1 && ret == 0)
    {
        pTmp = pTmp2;
        if (pTmp != NULL)
        {
            for (unsigned int i = 0; i < orderstatus_list.size(); ++i)
            {
                if (memcmp(&pTmp[0], &orderstatus_list[i].OrderNo[0], 5) == 0)
                {
                    ret = i + 1;
                    break;
                }
            }
        }
    }

    return ret;
}

void *TaifexOrderSession::Process()
{
    void *packet = NULL;

    if (r02_.SequentialGet(&packet))
    {
        return packet;
    }

    if (r03_.SequentialGet(&packet))
    {
        return packet;
    }
    return nullptr;
}

int TaifexOrderSession::GetSessionID()
{
    return session_id_;
}

}  // namespace alphaone
