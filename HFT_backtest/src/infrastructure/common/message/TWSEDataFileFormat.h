
#ifndef _TWSE_DATA_FILE_FORMAT_H_
#define _TWSE_DATA_FILE_FORMAT_H_

namespace alphaone
{

#pragma pack(1)

struct OrderReport63
{
    char OrderDate[8];
    char SecuritiesCode[6];
    char BuySell[1];
    char TradeTypeCode[1];
    char OrderTime[12];
    char OrderNumber2[5];
    char ChangedTradeCode[1];
    char OrderPrice[7];
    char ChangedtheTradeVolume[11];
    char OrderTypeCode[1];
    char NotesofInvestorsOrderChannel[1];
    char TypeofPrice[1];
    char TimeRestriction[1];
    char TypeofOrderData[1];
    char Filler[1];
    char TypeofInvestor[1];
    char OrderNumber1[4];
};

struct OrderReport59
{
    char OrderDate[8];
    char SecuritiesCode[6];
    char BuySell[1];
    char TradeTypeCode[1];
    char OrderTime[8];
    char OrderNumber2[5];
    char ChangedTradeCode[1];
    char OrderPrice[7];
    char ChangedtheTradeVolume[11];
    char OrderTypeCode[1];
    char NotesofInvestorsOrderChannel[1];
    char OrderReportPrint[4];
    char TypeofInvestor[1];
    char OrderNumber1[4];
};

struct MatchReport67
{
    char Date[8];
    char SecuritiesCode[6];
    char BuySell[1];
    char TradeTypeCode[1];
    char TradeTime[12];
    char TradeNumber[8];
    char OrderNumber2[5];
    char TradePrice[7];
    char TradeVolume[9];
    char TypeofPrice[1];
    char TimeRestriction[1];
    char Filler[2];
    char OrderTypeCode[1];
    char TypeofInvestor[1];
    char OrderNumber1[4];
};

struct MatchReport63
{
    char Date[8];
    char SecuritiesCode[6];
    char BuySell[1];
    char TradeTypeCode[1];
    char TradeTime[8];
    char TradeNumber[8];
    char OrderNumber2[5];
    char TradePrice[7];
    char TradeVolume[9];
    char TradingReportPrint[4];
    char OrderTypeCode[1];
    char TypeofInvestor[1];
    char OrderNumber1[4];
};

#pragma pack()

}  // namespace alphaone

#endif
