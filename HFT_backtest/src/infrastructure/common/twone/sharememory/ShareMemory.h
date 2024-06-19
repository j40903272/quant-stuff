#ifndef _SHAREMEMORY_H_
#define _SHAREMEMORY_H_

#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
namespace twone
{
class ShareMemory
{
  public:
    ShareMemory();
    ~ShareMemory();
    static void *Create(key_t key, int size, int *pIsCreate);
    static bool  IsExist(key_t key);
};
}  // namespace twone
#endif