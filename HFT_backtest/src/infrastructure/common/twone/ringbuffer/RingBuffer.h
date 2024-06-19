#ifndef _RINGBUFFER_H_
#define _RINGBUFFER_H_
namespace twone
{
struct RingBuffer_IndexBlock
{
    volatile int ReadyIndex;
    volatile int NextIndex;
};

class RingBuffer
{
  public:
    RingBuffer();
    RingBuffer(int indexKey, int memoryblockKey, int packetsize, int packetcount,
               int defaultReadyIndex, int defaultNextIndex);
    ~RingBuffer();

    RingBuffer(const RingBuffer &another);
    RingBuffer &operator=(const RingBuffer &another);

    void *GetNextAddress();
    void *GetAddress(int index);
    void *GetLastestReadyAddress();
    bool  SequentialGet(void **pDataBuffer);
    void *SequentialGetLastestAddress();

    int   GetReadyIndex();
    int   GetNextIndex();
    void  AddReadyIndex(int value);
    void  AddNextIndex(int value);
    int   GetPacketSize();
    void  Copy(int srcIndex, int dstIndex);
    void  SetUserDefine(void *userdefine);
    void *GetUserDefine();
    int   GetSequentialGetNextDoIndex();
    void  SetNextDoIndex(int index);

  private:
    int                             m_DataIndexKey;
    int                             m_DataBlockKey;
    int                             m_PacketSize;
    int                             m_PacketCount;
    int                             m_ModNum;
    int                             m_NextDoIndex;
    volatile RingBuffer_IndexBlock *m_pIndexBlock;
    volatile void *                 m_pMemoryBlock;
    void *                          m_pUserDefine;
};
}  // namespace twone
#endif