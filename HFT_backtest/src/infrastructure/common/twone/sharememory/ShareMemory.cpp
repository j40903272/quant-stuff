#include "ShareMemory.h"

#include <stdexcept>

namespace twone
{
ShareMemory::ShareMemory()
{
}

ShareMemory::~ShareMemory()
{
}

void *ShareMemory::Create(key_t key, int size, int *pIsCreate)
{
    void *shm_addr = NULL;

    int  shm_id = shmget(key, size, IPC_EXCL | 0666);
    char error[128];

    if (shm_id >= 0)
    {
        *pIsCreate = 0;
    }
    else
    {
        shm_id = shmget(key, size, IPC_CREAT | 0666);
        if (shm_id < 0)
        {
            sprintf(error, "[shmget] key=%d size=%d\n", key, size);
            throw std::runtime_error(error);
        }
        else
        {
            *pIsCreate = 1;
        }
    }

    if ((shm_addr = shmat(shm_id, NULL, 0)) == (char *)-1)
    {
        sprintf(error, "[shmat] key=%d \n", key);
        throw std::runtime_error(error);
    }

    struct shmid_ds buf;
    int             error_code = shmctl(shm_id, IPC_STAT, &buf);
    if (error_code < 0)
    {
        sprintf(error, "[shmctl] Cannot access shared memory information for segment %d key %d\n",
                shm_id, key);
        throw std::runtime_error(error);
    }
    else if (buf.shm_segsz != (size_t)size)
    {
        sprintf(error, "ringbuffer key=%d id=%d shm_segsz=%lu size=%d\n", key, shm_id,
                buf.shm_segsz, size);
        throw std::runtime_error(error);
    }

    shmctl(shm_id, SHM_LOCK, 0);

    return shm_addr;
}

bool ShareMemory::IsExist(key_t key)
{
    int ret = shmget(key, 0, 0);
    if (ret < 0)
    {
        return false;
    }
    return true;
}
}  // namespace twone