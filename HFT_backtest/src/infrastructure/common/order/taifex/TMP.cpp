#include "infrastructure/common/order/taifex/TMP.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

namespace alphaone
{
void TMPSetCheckSum(unsigned char *Checksum, const void *Data, size_t DataLen)
{
    char *       ptr = (char *)Data;
    char *       End = ptr + DataLen - 1;
    unsigned int Sum = 0;

    for (; ptr != End; (Sum += (unsigned char)(*(ptr++))))
        ;

    (*Checksum) = (unsigned char)(Sum & 255);

    return;
}


void TMPHdrSet(TMPHdr_t *Hdr, unsigned char MsgType, size_t MsgLen)
{
    TMPSET_MSGLEN(Hdr->msg_length, MsgLen);
    TMPSET_UINT8(Hdr->MessageType, MsgType);
    TMPSET_INT32(Hdr->msg_time.epoch_s, time(NULL));

    return;
}

TMPL10_t MakeL10(unsigned short fcm_id, unsigned short session_id)
{
    TMPL10_t packet;
    memset(&packet, 0, sizeof(TMPL10_t));

    TMPHdrSet(&packet.hdr, 10, sizeof(TMPL10_t));
    TMPSET_UINT32(packet.hdr.MsgSeqNum, 0);
    TMPSET_UINT16(packet.hdr.fcm_id, fcm_id);
    TMPSET_UINT16(packet.hdr.session_id, session_id);
    TMPSET_UINT8(packet.status_code, 0);
    TMPSET_UINT32(packet.start_in_bound_num, 0);

    TMPSetCheckSum(&packet.CheckSum, &packet, sizeof(TMPL10_t));

    return packet;
}

TMPL20_t MakeL20(unsigned short fcm_id, unsigned short session_id)
{
    TMPL20_t packet;
    memset(&packet, 0, sizeof(TMPL20_t));

    TMPHdrSet(&packet.hdr, 20, sizeof(TMPL20_t));
    TMPSET_UINT32(packet.hdr.MsgSeqNum, 0);
    TMPSET_UINT16(packet.hdr.fcm_id, fcm_id);
    TMPSET_UINT16(packet.hdr.session_id, session_id);
    TMPSET_UINT8(packet.status_code, 0);

    TMPSetCheckSum(&packet.CheckSum, &packet, sizeof(TMPL20_t));

    return packet;
}

TMPL40_t MakeL40(unsigned short fcm_id, unsigned short session_id, unsigned char UChar,
                 unsigned int AppendNo, int end_out_bound_num, int password)
{
    TMPL40_t packet;
    memset(&packet, 0, sizeof(TMPL40_t));

    TMPHdrSet(&packet.hdr, 40, sizeof(TMPL40_t));
    TMPSET_UINT16(packet.hdr.fcm_id, fcm_id);
    TMPSET_UINT16(packet.hdr.session_id, session_id);

    TMPSET_UINT8(packet.status_code, 0);

    TMPSET_UINT16(packet.fcm_id, fcm_id);
    TMPSET_UINT16(packet.session_id, session_id);
    TMPSET_UINT8(packet.ap_code, 4);
    TMPSET_UINT8(packet.cancel_order_sec, 30); /* order delay 30 sec, reject it */
    TMPSET_UINT8(packet.system_type, UChar);
    TMPSET_UINT16(packet.append_no, AppendNo);
    TMPSET_UINT32(packet.request_start_seq, end_out_bound_num);
    printf("pwd=%d, appendno=%d, res=%d\n", password, AppendNo,
           ((AppendNo * password) % 10000) / 100);
    TMPSET_UINT8(packet.key_value, ((AppendNo * password) % 10000) / 100);

    TMPSetCheckSum(&packet.CheckSum, &packet, sizeof(TMPL40_t));

    return packet;
}

TMPL42_t MakeL42(unsigned short fcm_id, unsigned short session_id)
{
    TMPL42_t packet;
    memset(&packet, 0, sizeof(TMPL42_t));

    TMPHdrSet(&packet.hdr, 42, sizeof(TMPL42_t));

    TMPSET_UINT32(packet.hdr.MsgSeqNum, 0);
    TMPSET_UINT16(packet.hdr.fcm_id, fcm_id);
    TMPSET_UINT16(packet.hdr.session_id, session_id);
    TMPSET_UINT8(packet.status_code, 0);
    TMPSetCheckSum(&packet.CheckSum, &packet, sizeof(TMPL42_t));

    return packet;
}


TMPL60_t MakeL60(unsigned short fcm_id, unsigned short session_id)
{
    TMPL60_t packet;
    memset(&packet, 0, sizeof(TMPL60_t));

    TMPHdrSet(&packet.hdr, 60, sizeof(TMPL60_t));

    TMPSET_UINT32(packet.hdr.MsgSeqNum, 0);
    TMPSET_UINT16(packet.hdr.fcm_id, fcm_id);
    TMPSET_UINT16(packet.hdr.session_id, session_id);
    TMPSET_UINT8(packet.status_code, 0);
    TMPSetCheckSum(&packet.CheckSum, &packet, sizeof(TMPL60_t));

    return packet;
}


TMPR04_t MakeR04(unsigned short fcm_id, unsigned short session_id)
{
    TMPR04_t packet;
    memset(&packet, 0, sizeof(TMPR04_t));

    TMPHdrSet(&packet.hdr, TMPMsgType_R04, sizeof(TMPR04_t));
    TMPSET_UINT32(packet.hdr.MsgSeqNum, 0);
    TMPSET_UINT16(packet.hdr.fcm_id, fcm_id);
    TMPSET_UINT16(packet.hdr.session_id, session_id);
    TMPSET_UINT8(packet.status_code, 0);

    TMPSetCheckSum(&packet.CheckSum, &packet, sizeof(TMPR04_t));

    return packet;
}

TMPR05_t MakeR05(unsigned short fcm_id, unsigned short session_id)
{
    TMPR05_t packet;
    memset(&packet, 0, sizeof(TMPR05_t));

    TMPHdrSet(&packet.hdr, TMPMsgType_R05, sizeof(TMPR05_t));
    TMPSET_UINT32(packet.hdr.MsgSeqNum, 0);
    TMPSET_UINT16(packet.hdr.fcm_id, fcm_id);
    TMPSET_UINT16(packet.hdr.session_id, session_id);
    TMPSET_UINT8(packet.status_code, 0);

    TMPSetCheckSum(&packet.CheckSum, &packet, sizeof(TMPR05_t));

    return packet;
}

TMPR11_t MakeR11(unsigned short fcm_id, unsigned short session_id, unsigned short TradeReqID,
                 unsigned char flow_group_no, unsigned char SubscriptionRequestType)
{
    TMPR11_t packet;
    memset(&packet, 0, sizeof(TMPR11_t));

    TMPHdrSet(&packet.hdr, TMPMsgType_R11, sizeof(TMPR11_t));
    TMPSET_UINT32(packet.hdr.MsgSeqNum, 0);
    TMPSET_UINT16(packet.hdr.fcm_id, fcm_id);
    TMPSET_UINT16(packet.hdr.session_id, session_id);
    TMPSET_UINT8(packet.status_code, 0);
    TMPSET_UINT16(packet.TradeReqID, TradeReqID);
    TMPSET_UINT8(packet.flow_group_no, flow_group_no);
    TMPSET_UINT8(packet.SubscriptionRequestType, SubscriptionRequestType);


    TMPSetCheckSum(&packet.CheckSum, &packet, sizeof(TMPR11_t));

    return packet;
}
}  // namespace alphaone
