#include "Order.h"

#include <string>
namespace alphaone
{

const int CharToIntMappingTable[123] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  0,  0,  0,  0,  0,
    0,  0,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 0,  0,  0,  0,  0,  0,  36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61};

const char IntToCharMappingTable[62] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

const int multiply[5] = {1, 62, 62 * 62, 62 * 62 * 62, 62 * 62 * 62 * 62};

// 0->9->A->Z->a->z->0
int OrderNo_NextFn(char *CurSeqNo)
{
    char *ptr;
    ptr = CurSeqNo + 4;

    while (1)
    {
        switch (*ptr)
        {
        case 'z':
            *ptr = '0';
            --ptr;
            continue;
            break;
        case '9':
            *ptr = 'A';
            break;
        case 'Z':
            *ptr = 'a';
            break;
        default:
            ++(*ptr);
            break;
        }

        break;
    }

    return 0;
}

int OrderNo_NextFn_WithLength(char *CurSeqNo, int length)
{
    char *ptr;
    ptr = CurSeqNo + length - 1;

    while (1)
    {
        switch (*ptr)
        {
        case 'z':
            *ptr = '0';
            --ptr;
            continue;
            break;
        case '9':
            *ptr = 'A';
            break;
        case 'Z':
            *ptr = 'a';
            break;
        default:
            ++(*ptr);
            break;
        }

        break;
    }

    return 0;
}

int OrderNoToInt(const char *orderNo)
{
    char *ptr = const_cast<char *>(orderNo + 4);
    int   res{0}, index{0}, i{0};
    for (; i < 5; ++i)
    {
        char c = *(ptr--);
        index  = static_cast<int>(c);
        res += CharToIntMappingTable[index] * multiply[i];
    }
    return res;
}

int OrderNoToInt_WithLength(const char *orderNo, int length)
{
    char *ptr = const_cast<char *>(orderNo + length - 1);
    int   res{0}, i{0}, index{0};
    for (; i < length; ++i)
    {
        char c = *(ptr--);
        index  = static_cast<int>(c);
        res += CharToIntMappingTable[index] * multiply[i];
    }
    return res;
}

std::string IntToOrderNo(int number)
{
    if (number < 0)
    {
        return "";
    }
    char orderno[5] = {'0', '0', '0', '0', '0'};
    int  q;
    int  r;
    int  idx = 4;
    while ((q = number / 62) != 0)
    {
        r            = number % 62;
        orderno[idx] = IntToCharMappingTable[r];
        idx--;
        number = q;
    }

    orderno[idx] = IntToCharMappingTable[number];
    return std::string(orderno, 5);
}
}  // namespace alphaone