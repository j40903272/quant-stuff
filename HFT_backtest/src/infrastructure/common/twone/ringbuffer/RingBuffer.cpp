#include "RingBuffer.h"

#include "../sharememory/ShareMemory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
namespace twone
{
RingBuffer::RingBuffer()
{
    this->m_pIndexBlock  = NULL;
    this->m_pMemoryBlock = NULL;

    m_pUserDefine  = NULL;
    m_DataIndexKey = 0;
    m_DataBlockKey = 0;
    m_PacketSize   = 0;
    m_PacketCount  = 0;
}

RingBuffer::RingBuffer(const RingBuffer &another)
{
    m_DataIndexKey = another.m_DataIndexKey;
    m_DataBlockKey = another.m_DataBlockKey;
    m_PacketSize   = another.m_PacketSize;
    m_PacketCount  = another.m_PacketCount;
    m_ModNum       = another.m_ModNum;
    m_NextDoIndex  = another.m_NextDoIndex;
    m_pUserDefine  = another.m_pUserDefine;

    if (another.m_DataIndexKey == 0)
    {
        m_pIndexBlock             = new RingBuffer_IndexBlock();
        m_pIndexBlock->ReadyIndex = another.m_pIndexBlock->ReadyIndex;
        m_pIndexBlock->NextIndex  = another.m_pIndexBlock->NextIndex;
    }
    else
    {
        m_pIndexBlock = another.m_pIndexBlock;
    }

    if (another.m_DataBlockKey == 0)
    {
        m_pMemoryBlock = new char[m_PacketSize * m_PacketCount];
        memset((void *)m_pMemoryBlock, 0, m_PacketSize * m_PacketCount);
    }
    else
    {
        m_pMemoryBlock = another.m_pMemoryBlock;
    }
}

RingBuffer &RingBuffer::operator=(const RingBuffer &another)
{
    m_DataIndexKey = another.m_DataIndexKey;
    m_DataBlockKey = another.m_DataBlockKey;
    m_PacketSize   = another.m_PacketSize;
    m_PacketCount  = another.m_PacketCount;
    m_ModNum       = another.m_ModNum;
    m_NextDoIndex  = another.m_NextDoIndex;
    m_pUserDefine  = another.m_pUserDefine;

    if (another.m_DataIndexKey == 0)
    {
        m_pIndexBlock             = new RingBuffer_IndexBlock();
        m_pIndexBlock->ReadyIndex = another.m_pIndexBlock->ReadyIndex;
        m_pIndexBlock->NextIndex  = another.m_pIndexBlock->NextIndex;
    }
    else
    {
        m_pIndexBlock = another.m_pIndexBlock;
    }

    if (another.m_DataBlockKey == 0)
    {
        m_pMemoryBlock = new char[m_PacketSize * m_PacketCount];
    }
    else
    {
        m_pMemoryBlock = another.m_pMemoryBlock;
    }
    return *this;
}

RingBuffer::RingBuffer(int indexKey, int memoryblockKey, int packetsize, int packetcount,
                       int defaultReadyIndex, int defaultNextIndex)
{
    m_DataIndexKey = indexKey;
    m_DataBlockKey = memoryblockKey;
    m_PacketSize   = packetsize;
    m_PacketCount  = packetcount;
    m_ModNum       = packetcount - 1;
    m_pUserDefine  = NULL;

    int isCreate = 0;
    if (indexKey == 0)
    {
        this->m_pIndexBlock = new RingBuffer_IndexBlock();
        isCreate            = 1;
    }
    else
    {
        this->m_pIndexBlock = (RingBuffer_IndexBlock *)ShareMemory::Create(
            indexKey, sizeof(RingBuffer_IndexBlock), &isCreate);
    }

    if (isCreate == 1)
    {
        this->m_pIndexBlock->ReadyIndex = defaultReadyIndex;
        this->m_pIndexBlock->NextIndex  = defaultNextIndex;
    }
    else
    {
        if (this->m_pIndexBlock->ReadyIndex == 0 && this->m_pIndexBlock->NextIndex == 0)
        {
            this->m_pIndexBlock->ReadyIndex = defaultReadyIndex;
            this->m_pIndexBlock->NextIndex  = defaultNextIndex;
        }

        if (indexKey >= 10000 && indexKey <= 89999)  // for order
        {
            if (this->m_pIndexBlock->ReadyIndex == this->m_pIndexBlock->NextIndex)
            {
                printf("this->m_pIndexBlock->ReadyIndex=%d, this->m_pIndexBlock->NextIndex=%d, "
                       "indexKey=%d, memoryblockKey=%d\n",
                       this->m_pIndexBlock->ReadyIndex, this->m_pIndexBlock->NextIndex, indexKey,
                       memoryblockKey);
                exit(0);
            }
            else if (this->m_pIndexBlock->NextIndex - this->m_pIndexBlock->ReadyIndex != 1)
            {
                int maxindex = this->m_pIndexBlock->NextIndex > this->m_pIndexBlock->ReadyIndex
                                   ? this->m_pIndexBlock->NextIndex
                                   : this->m_pIndexBlock->ReadyIndex;

                this->m_pIndexBlock->ReadyIndex = maxindex + 1;
                this->m_pIndexBlock->NextIndex  = maxindex + 2;
            }
        }
    }

    if (memoryblockKey == 0)
    {
        this->m_pMemoryBlock = new char[m_PacketSize * m_PacketCount];
    }
    else
    {
        this->m_pMemoryBlock =
            ShareMemory::Create(memoryblockKey, m_PacketSize * m_PacketCount, &isCreate);
    }

    m_NextDoIndex = this->GetReadyIndex() + 1;
}

RingBuffer::~RingBuffer()
{
    if (m_DataIndexKey == 0)
    {
        if (this->m_pIndexBlock != NULL)
        {
            delete this->m_pIndexBlock;
        }
    }

    if (m_DataBlockKey == 0)
    {
        if (this->m_pMemoryBlock != NULL)
        {
            delete[](char *) this->m_pMemoryBlock;
        }
    }
}

void *RingBuffer::GetNextAddress()
{
    return this->GetAddress(m_pIndexBlock->NextIndex);
}

int RingBuffer::GetReadyIndex()
{
    return this->m_pIndexBlock->ReadyIndex;
}

int RingBuffer::GetNextIndex()
{
    return this->m_pIndexBlock->NextIndex;
}

void RingBuffer::AddReadyIndex(int value)
{
    auto& readyIndex = this->m_pIndexBlock->ReadyIndex; // Create a reference to avoid multiple access
    auto currentValue = readyIndex; // Read once, into a non-volatile temporary
    currentValue += value; // Do the arithmetic operation on the non-volatile temporary
    readyIndex = currentValue; // Write the result back in a single operation

    // this->m_pIndexBlock->ReadyIndex += value;
}

void RingBuffer::AddNextIndex(int value)
{
    auto& nextIndex = this->m_pIndexBlock->NextIndex; // Create a reference to avoid multiple access
    auto currentValue = nextIndex; // Read once, into a non-volatile temporary
    currentValue += value; // Do the arithmetic operation on the non-volatile temporary
    nextIndex = currentValue; // Write the result back in a single operation

    // this->m_pIndexBlock->NextIndex += value;
}

void *RingBuffer::GetAddress(int index)
{
    int offset = (index & m_ModNum) * m_PacketSize;
    return (void *)((char *)this->m_pMemoryBlock + offset);
}

void *RingBuffer::GetLastestReadyAddress()
{
    return GetAddress(this->m_pIndexBlock->ReadyIndex);
}

int RingBuffer::GetPacketSize()
{
    return m_PacketSize;
}

void RingBuffer::Copy(int srcIndex, int dstIndex)
{
    void *src = GetAddress(srcIndex);
    void *dst = GetAddress(dstIndex);
    memcpy(dst, src, m_PacketSize);
}

bool RingBuffer::SequentialGet(void **pDataBuffer)
{
    if (m_NextDoIndex <= GetReadyIndex())
    {
        *pDataBuffer = GetAddress(m_NextDoIndex);
        m_NextDoIndex++;
        return true;
    }
    return false;
}

void *RingBuffer::SequentialGetLastestAddress()
{
    int readyIndex = GetReadyIndex();
    if (m_NextDoIndex <= readyIndex)
    {
        m_NextDoIndex = readyIndex + 1;
        return GetAddress(readyIndex);
    }
    return NULL;
}

int RingBuffer::GetSequentialGetNextDoIndex()
{
    return m_NextDoIndex;
}

void RingBuffer::SetUserDefine(void *userdefine)
{
    m_pUserDefine = userdefine;
}

void *RingBuffer::GetUserDefine()
{
    return m_pUserDefine;
}

void RingBuffer::SetNextDoIndex(int index)
{
    m_NextDoIndex = index;
}
}  // namespace twone