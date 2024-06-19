#include "infrastructure/common/twone/orderbook/TaifexOrderBook.h"

#include "infrastructure/common/typedef/Typedefs.h"

#include <algorithm>
#include <iterator>
#include <string.h>

namespace twone
{
TaifexOrderBook::TaifexOrderBook(const alphaone::Symbol *symbol)
    : msg{alphaone::DataSourceType::MarketByPrice}, last_prod_seq_(0), symbol_(symbol)
{
    std::fill(std::begin(msg.mbp.bid_price), std::end(msg.mbp.bid_price), alphaone::EMPTY_BID);
    std::fill(std::begin(msg.mbp.ask_price), std::end(msg.mbp.ask_price), alphaone::EMPTY_ASK);

    std::fill(std::begin(msg.mbp.bid_qty), std::end(msg.mbp.bid_qty), 0);
    std::fill(std::begin(msg.mbp.ask_qty), std::end(msg.mbp.ask_qty), 0);

    std::fill(std::begin(msg.implied.bid_price), std::end(msg.implied.bid_price),
              alphaone::EMPTY_BID);

    std::fill(std::begin(msg.implied.ask_price), std::end(msg.implied.ask_price),
              alphaone::EMPTY_ASK);

    std::fill(std::begin(msg.implied.bid_qty), std::end(msg.implied.bid_qty), 0);
    std::fill(std::begin(msg.implied.ask_qty), std::end(msg.implied.ask_qty), 0);

    msg.symbol          = symbol;
    msg.mbp.count       = 5;
    msg.exchange_time   = alphaone::Timestamp::from_epoch_nsec(0);
    msg.sequence_number = 0;

    msg.implied.count         = 1;
    msg.implied.is_packet_end = true;
}

TaifexOrderBook::~TaifexOrderBook()
{
}

TaifexOrderBook::TaifexOrderBook(const TaifexOrderBook &another)
    : msg{alphaone::DataSourceType::MarketByPrice}
{
    memcpy(msg.mbp.bid_price, another.msg.mbp.bid_price, 10 * sizeof(double));
    memcpy(msg.mbp.bid_qty, another.msg.mbp.bid_qty, 10 * sizeof(double));
    memcpy(msg.mbp.ask_price, another.msg.mbp.ask_price, 10 * sizeof(double));
    memcpy(msg.mbp.ask_qty, another.msg.mbp.ask_qty, 10 * sizeof(double));

    memcpy(msg.implied.bid_price, another.msg.implied.bid_price, 5 * sizeof(double));
    memcpy(msg.implied.bid_qty, another.msg.implied.bid_qty, 5 * sizeof(double));
    memcpy(msg.implied.ask_price, another.msg.implied.ask_price, 5 * sizeof(double));
    memcpy(msg.implied.ask_qty, another.msg.implied.ask_qty, 5 * sizeof(double));

    last_prod_seq_ = another.last_prod_seq_;
    map_packet_    = another.map_packet_;
    symbol_        = another.symbol_;
}

TaifexOrderBook &TaifexOrderBook::operator=(const TaifexOrderBook &another)
{
    memcpy(msg.mbp.bid_price, another.msg.mbp.bid_price, 10 * sizeof(double));
    memcpy(msg.mbp.bid_qty, another.msg.mbp.bid_qty, 10 * sizeof(double));
    memcpy(msg.mbp.ask_price, another.msg.mbp.ask_price, 10 * sizeof(double));
    memcpy(msg.mbp.ask_qty, another.msg.mbp.ask_qty, 10 * sizeof(double));

    memcpy(msg.implied.bid_price, another.msg.implied.bid_price, 5 * sizeof(double));
    memcpy(msg.implied.bid_qty, another.msg.implied.bid_qty, 5 * sizeof(double));
    memcpy(msg.implied.ask_price, another.msg.implied.ask_price, 5 * sizeof(double));
    memcpy(msg.implied.ask_qty, another.msg.implied.ask_qty, 5 * sizeof(double));

    last_prod_seq_ = another.last_prod_seq_;
    map_packet_    = another.map_packet_;
    symbol_        = another.symbol_;
    return *this;
}

void TaifexOrderBook::ProcessRewind(alphaone::Timestamp &ts)
{
    int beginSeq = last_prod_seq_ + 1;

    while (true)
    {
        auto iter = map_packet_.find(beginSeq);
        if (iter == map_packet_.end())  // not found
        {
            break;
        }

        alphaone::TXMarketDataHdr_RealTime_t *pData =
            (alphaone::TXMarketDataHdr_RealTime_t *)iter->second;
        if (pData->MessageKind[0] == 'A')  // I081
        {
            ParseI081((alphaone::TXMarketDataI081_t *)pData, false, ts);
        }
        else if (pData->MessageKind[0] == 'D')  // I024
        {
            ParseI024((alphaone::TXMarketDataI024_t *)pData, false, ts);
        }
        else if (pData->MessageKind[0] == 'B')  // I083
        {
            ParseI083((alphaone::TXMarketDataI083_t *)pData, false, ts);
        }
        else if (pData->MessageKind[0] == 'E')  // I025
        {
            ParseI025((alphaone::TXMarketDataI025_t *)pData, false, ts);
        }
        beginSeq++;
    }
}

TaifexUpdateFlag TaifexOrderBook::ParseI081(alphaone::TXMarketDataI081_t *pI081, bool isrewind,
                                            alphaone::Timestamp &ts)
{
    TaifexUpdateFlag ret = TaifexUpdateFlag::Invalid;

    msg.provider_time = ts;
    msg.exchange_time = alphaone::Timestamp::from_epoch_nsec(0);

    // SeqNo
    int seqno = Decode5(pI081->ProductMsgSeq);
    if (seqno - last_prod_seq_ != 1)  // jump, save packet
    {
        map_packet_[seqno] = (void *)pI081;
        return TaifexUpdateFlag::NoChange;
    }

    // Parse
    int noEntries = Decode1(pI081->NO_MD_ENTRIES);
    for (int i = 0; i < noEntries; ++i)
    {
        char action    = pI081->FurtherInfo[i].MD_UPDATE_ACTION[0];
        char entryType = pI081->FurtherInfo[i].MD_ENTRY_TYPE[0];
        int  sign      = (pI081->FurtherInfo[i].Sign[0] == '0') ? 1 : -1;
        int  price     = Decode5(pI081->FurtherInfo[i].Price) * sign;
        int  qty       = Decode4(pI081->FurtherInfo[i].Qty);
        int  level     = Decode1(pI081->FurtherInfo[i].PriceLevel);  // 1,2,3,4,5

        int index = level - 1;

        if (action == '0')  // New
        {
            if (entryType == '0')  // Buy
            {
                if (msg.mbp.bid_price[index] == alphaone::EMPTY_BID)
                {
                    msg.mbp.bid_price[index] = price;
                    msg.mbp.bid_qty[index]   = qty;
                }
                else
                {
                    AddBid(level, price, qty);
                }
            }
            else if (entryType == '1')  // Sell
            {
                if (msg.mbp.ask_price[index] == alphaone::EMPTY_ASK)
                {
                    msg.mbp.ask_price[index] = price;
                    msg.mbp.ask_qty[index]   = qty;
                }
                else
                {
                    AddAsk(level, price, qty);
                }
            }
            ret |= TaifexUpdateFlag::OrderBook;
        }
        else if (action == '1')  // Change
        {
            if (entryType == '0')  // Buy
            {
                msg.mbp.bid_price[index] = price;
                msg.mbp.bid_qty[index]   = qty;
            }
            else if (entryType == '1')  // Sell
            {
                msg.mbp.ask_price[index] = price;
                msg.mbp.ask_qty[index]   = qty;
            }
            ret |= TaifexUpdateFlag::OrderBook;
        }
        else if (action == '2')  // Delete
        {
            if (entryType == '0')  // Buy
            {
                DeleteBid(level);
            }
            else if (entryType == '1')  // Sell
            {
                DeleteAsk(level);
            }
            ret |= TaifexUpdateFlag::OrderBook;
        }
        else if (action == '5')
        {
            if (entryType == 'E')  // Implied Bid
            {
                if (qty > 0)
                {
                    msg.implied.bid_price[0] = price;
                    msg.implied.bid_qty[0]   = qty;
                }
                else
                {
                    msg.implied.bid_price[0] = alphaone::EMPTY_BID;
                    msg.implied.bid_qty[0]   = qty;
                }
            }
            else if (entryType == 'F')  // Implied Ask
            {
                if (qty > 0)
                {
                    msg.implied.ask_price[0] = price;
                    msg.implied.ask_qty[0]   = qty;
                }
                else
                {
                    msg.implied.ask_price[0] = alphaone::EMPTY_ASK;
                    msg.implied.ask_qty[0]   = qty;
                }
            }

            ret |= TaifexUpdateFlag::ImpliedBook;
        }
    }

    last_prod_seq_        = seqno;
    msg.mbp.is_packet_end = true;

    if (isrewind)
    {
        ProcessRewind(ts);
    }

    return ret;
}

TaifexUpdateFlag TaifexOrderBook::ParseI083(alphaone::TXMarketDataI083_t *pI083, bool isrewind,
                                            alphaone::Timestamp &ts)
{
    TaifexUpdateFlag ret = TaifexUpdateFlag::Invalid;
    msg.provider_time    = ts;
    msg.exchange_time    = alphaone::Timestamp::from_epoch_nsec(0);

    // SeqNo
    int seqno = Decode5(pI083->ProductMsgSeq);
    if (seqno <= last_prod_seq_)  // error drop
    {
        return ret;
    }

    std::fill(std::begin(msg.mbp.bid_price), std::end(msg.mbp.bid_price), alphaone::EMPTY_BID);
    std::fill(std::begin(msg.mbp.ask_price), std::end(msg.mbp.ask_price), alphaone::EMPTY_ASK);

    std::fill(std::begin(msg.mbp.bid_qty), std::end(msg.mbp.bid_qty), 0);
    std::fill(std::begin(msg.mbp.ask_qty), std::end(msg.mbp.ask_qty), 0);

    std::fill(std::begin(msg.implied.bid_price), std::end(msg.implied.bid_price),
              alphaone::EMPTY_BID);

    std::fill(std::begin(msg.implied.ask_price), std::end(msg.implied.ask_price),
              alphaone::EMPTY_ASK);

    std::fill(std::begin(msg.implied.bid_qty), std::end(msg.implied.bid_qty), 0);
    std::fill(std::begin(msg.implied.ask_qty), std::end(msg.implied.ask_qty), 0);

    // Parse
    int noEntries = Decode1(pI083->NO_MD_ENTRIES);

    for (int i = 0; i < noEntries; ++i)
    {
        char entryType = pI083->FurtherInfo[i].MD_ENTRY_TYPE[0];
        int  sign      = pI083->FurtherInfo[i].Sign[0] == '0' ? 1 : -1;
        int  price     = Decode5(pI083->FurtherInfo[i].Price) * sign;
        int  qty       = Decode4(pI083->FurtherInfo[i].Qty);
        int  level     = Decode1(pI083->FurtherInfo[i].PriceLevel);  // 1,2,3,4,5

        int index = level - 1;
        if (entryType == '0')  // Buy
        {
            msg.mbp.bid_price[index] = price;
            msg.mbp.bid_qty[index]   = qty;
            ret |= TaifexUpdateFlag::OrderBook;
        }
        else if (entryType == '1')  // Sell
        {
            msg.mbp.ask_price[index] = price;
            msg.mbp.ask_qty[index]   = qty;
            ret |= TaifexUpdateFlag::OrderBook;
        }
        else if (entryType == 'E')  // Implied Bid
        {
            if (qty > 0)
            {
                msg.implied.bid_price[0] = price;
                msg.implied.bid_qty[0]   = qty;
            }
            else
            {
                msg.implied.bid_price[0] = alphaone::EMPTY_BID;
                msg.implied.bid_qty[0]   = qty;
            }
            ret |= TaifexUpdateFlag::ImpliedBook;
        }
        else if (entryType == 'F')  // Implied Ask
        {
            if (qty > 0)
            {
                msg.implied.ask_price[0] = price;
                msg.implied.ask_qty[0]   = qty;
            }
            else
            {
                msg.implied.ask_price[0] = alphaone::EMPTY_ASK;
                msg.implied.ask_qty[0]   = qty;
            }
            ret |= TaifexUpdateFlag::ImpliedBook;
        }
    }

    last_prod_seq_        = seqno;
    msg.mbp.is_packet_end = true;

    if (isrewind)
    {
        ProcessRewind(ts);
    }

    return ret;
}

void TaifexOrderBook::ParseI084(alphaone::TXMarketDataI084_O_Info_t *pI084, int lastProdSeq)
{
    int level = Decode1(pI084->PriceLevel) - 1;
    int sign  = (pI084->Sign[0] == '0') ? 1 : -1;
    int price = Decode5(pI084->Price) * sign;
    int qty   = Decode4(pI084->Qty);

    if (pI084->MD_ENTRY_TYPE[0] == '0')  //買
    {
        msg.mbp.bid_price[level] = price;
        msg.mbp.bid_qty[level]   = qty;
    }
    else if (pI084->MD_ENTRY_TYPE[0] == '1')  //賣
    {
        msg.mbp.ask_price[level] = price;
        msg.mbp.ask_qty[level]   = qty;
    }

    last_prod_seq_ = lastProdSeq;
}

void TaifexOrderBook::ParseI024(alphaone::TXMarketDataI024_t *pI024, bool isrewind,
                                alphaone::Timestamp &ts)
{
    // SeqNo
    int seqno = Decode5(pI024->ProductMsgSeq);
    if (seqno - last_prod_seq_ != 1)  // jump, save packet
    {
        map_packet_[seqno] = (void *)pI024;
        return;
    }

    last_prod_seq_ = seqno;

    if (isrewind)
    {
        ProcessRewind(ts);
    }
}

void TaifexOrderBook::ParseI025(alphaone::TXMarketDataI025_t *pI025, bool isrewind,
                                alphaone::Timestamp &ts)
{
    // SeqNo
    int seqno = Decode5(pI025->ProductMsgSeq);
    if (seqno - last_prod_seq_ != 1)  // jump, save packet
    {
        map_packet_[seqno] = (void *)pI025;
        return;
    }

    last_prod_seq_ = seqno;

    if (isrewind)
    {
        ProcessRewind(ts);
    }
}

void TaifexOrderBook::Reset()
{
    std::fill(std::begin(msg.mbp.bid_price), std::end(msg.mbp.bid_price), alphaone::EMPTY_BID);
    std::fill(std::begin(msg.mbp.ask_price), std::end(msg.mbp.ask_price), alphaone::EMPTY_ASK);

    std::fill(std::begin(msg.mbp.bid_qty), std::end(msg.mbp.bid_qty), 0);
    std::fill(std::begin(msg.mbp.ask_qty), std::end(msg.mbp.ask_qty), 0);

    std::fill(std::begin(msg.implied.bid_price), std::end(msg.implied.bid_price),
              alphaone::EMPTY_BID);

    std::fill(std::begin(msg.implied.ask_price), std::end(msg.implied.ask_price),
              alphaone::EMPTY_ASK);

    std::fill(std::begin(msg.implied.bid_qty), std::end(msg.implied.bid_qty), 0);
    std::fill(std::begin(msg.implied.ask_qty), std::end(msg.implied.ask_qty), 0);

    last_prod_seq_ = 0;
    map_packet_.clear();
}

void TaifexOrderBook::AddBid(int level, int price, int qty)
{
    int tLevel = level - 1;
    if (level < 5)
    {
        memmove(&msg.mbp.bid_price[level], &msg.mbp.bid_price[tLevel],
                (5 - level) * sizeof(double));
        memmove(&msg.mbp.bid_qty[level], &msg.mbp.bid_qty[tLevel], (5 - level) * sizeof(double));
    }

    msg.mbp.bid_price[tLevel] = price;
    msg.mbp.bid_qty[tLevel]   = qty;
}

void TaifexOrderBook::AddAsk(int level, int price, int qty)
{
    int tLevel = level - 1;
    if (level < 5)
    {
        memmove(&msg.mbp.ask_price[level], &msg.mbp.ask_price[tLevel],
                (5 - level) * sizeof(double));
        memmove(&msg.mbp.ask_qty[level], &msg.mbp.ask_qty[tLevel], (5 - level) * sizeof(double));
    }

    msg.mbp.ask_price[tLevel] = price;
    msg.mbp.ask_qty[tLevel]   = qty;
}

void TaifexOrderBook::DeleteBid(int level)
{
    if (level < 5)
    {
        int tLevel = level - 1;
        memmove(&msg.mbp.bid_price[tLevel], &msg.mbp.bid_price[level],
                (5 - level) * sizeof(double));
        memmove(&msg.mbp.bid_qty[tLevel], &msg.mbp.bid_qty[level], (5 - level) * sizeof(double));
        msg.mbp.bid_price[4] = alphaone::EMPTY_BID;
        msg.mbp.bid_qty[4]   = 0;
    }
    else if (level == 5)
    {
        msg.mbp.bid_price[4] = alphaone::EMPTY_BID;
        msg.mbp.bid_qty[4]   = 0;
    }
}

void TaifexOrderBook::DeleteAsk(int level)
{
    if (level < 5)
    {
        int tLevel = level - 1;
        memmove(&msg.mbp.ask_price[tLevel], &msg.mbp.ask_price[level],
                (5 - level) * sizeof(double));
        memmove(&msg.mbp.ask_qty[tLevel], &msg.mbp.ask_qty[level], (5 - level) * sizeof(double));
        msg.mbp.ask_price[4] = alphaone::EMPTY_ASK;
        msg.mbp.ask_qty[4]   = 0;
    }
    else if (level == 5)
    {
        msg.mbp.ask_price[4] = alphaone::EMPTY_ASK;
        msg.mbp.ask_qty[4]   = 0;
    }
}

void TaifexOrderBook::Dump()
{
    for (int i = 0; i < 5; ++i)
    {
        printf("%lf\t%lf\t%lf\t%lf\n", msg.mbp.bid_qty[i], msg.mbp.bid_price[i],
               msg.mbp.ask_price[i], msg.mbp.ask_qty[i]);
    }
    printf("=======================Pid=%s, last_prod_seq_=%d================================\n",
           symbol_->GetDataSourcePid().c_str(), last_prod_seq_);
}
}  // namespace twone
