#ifndef _ORDER_H
#define _ORDER_H

#include <string>

namespace alphaone
{
// 0->9->A->Z->a->z->0
int OrderNo_NextFn(char *CurSeqNo);
int OrderNo_NextFn_WithLength(char *CurSeqNo, int length);

int OrderNoToInt(const char *orderNo);
int OrderNoToInt_WithLength(const char *orderNo, int length);

std::string IntToOrderNo(int number);

}  // namespace alphaone
#endif