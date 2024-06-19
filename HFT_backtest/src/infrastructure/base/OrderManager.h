#ifndef _ORDERMANAGER_H
#define _ORDERMANAGER_H

#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/util/PacketLog.h"

namespace alphaone
{

class OrderManager
{
  public:
    OrderManager() : packet_log_struct_(nullptr)
    {
    }

    virtual ~OrderManager() = default;

    virtual void Process(const Timestamp &event_loop_time) = 0;

    void SetPacketLogStruct(PacketLogStruct *packet_log_struct)
    {
        packet_log_struct_ = packet_log_struct;
    }

    PacketLogStruct *GetPacketLogStruct()
    {
        return packet_log_struct_;
    }

    void SaveOrderMapping(char *orderno, const char *userdefine, PacketLogType type,
                          PacketLogExecType ExecType)
    {
        if (packet_log_struct_ != nullptr)
        {
            if (packet_log_struct_->Type != (int)PacketLogType::INVALID)
            {
                packet_log_struct_->OrderTo  = (int)type;
                packet_log_struct_->ExecType = (char)ExecType;
                memcpy(packet_log_struct_->UserDefine, userdefine, 8);

                if (ExecType == PacketLogExecType::NEW)
                {
                    packet_log_struct_->OrderNo = orderno;
                }
                else if (ExecType == PacketLogExecType::CANCEL)
                {
                    memcpy(packet_log_struct_->OrderNoStr, orderno, 5);
                }

                void *data = (void *)packet_log_struct_->RingBuffer_PacketLog->GetNextAddress();
                memcpy(data, packet_log_struct_, sizeof(PacketLogStruct));

                packet_log_struct_->RingBuffer_PacketLog->AddReadyIndex(1);
                packet_log_struct_->RingBuffer_PacketLog->AddNextIndex(1);
                packet_log_struct_->SeqNum++;
            }
        }
    }

  private:
    PacketLogStruct *packet_log_struct_;
};

}  // namespace alphaone
#endif
