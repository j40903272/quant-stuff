#ifndef _POSITION_H_
#define _POSITION_H_

#include "infrastructure/base/Book.h"
#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/EnumToString.h"

namespace alphaone
{
struct Position
{
    Position(const GlobalConfiguration *configuration, const Symbol *symbol, const Book *book,
             int position_long = 0, int position_short = 0, int64_t investment_long = 0,
             int64_t investment_short = 0, unsigned int hit_count_long = 0,
             unsigned int hit_count_short = 0)
        : symbol_{symbol}
        , book_{book}
        , product_type_{symbol->GetProductType()}
        , multiplier_{symbol->GetMultiplier()}
        , json_(configuration->GetJson().at("Position"))
        , fee_rate_{json_["fee_rate"].get<double>()}
        , tax_rate_{json_.value("tax_rate", 0.)}
        , fee_cost_{json_["fee_cost"].get<double>()}
        , position_{position_short, position_long}
        , investment_{investment_short, investment_long}
        , hit_count_{hit_count_short, hit_count_long}
        , zero_cross_count_{0}
        , cost_{0.0}
    {
    }

    Position(const Symbol *symbol, const Book *book, const double fee_rate, const double fee_cost,
             const double tax_rate = 0.0, int position_long = 0, int position_short = 0,
             int64_t investment_long = 0, int64_t investment_short = 0,
             unsigned int hit_count_long = 0, unsigned int hit_count_short = 0)
        : symbol_{symbol}
        , book_{book}
        , product_type_{symbol->GetProductType()}
        , multiplier_{symbol->GetMultiplier()}
        , fee_rate_{fee_rate}
        , tax_rate_{tax_rate}
        , fee_cost_{fee_cost}
        , position_{position_short, position_long}
        , investment_{investment_short, investment_long}
        , hit_count_{hit_count_short, hit_count_long}
        , zero_cross_count_{0}
        , cost_{0.0}
    {
    }

    inline void Update(const BookSide side, const int64_t price, const int64_t qty,
                       bool should_update_cost = true)
    {
        const BookQty old_position{GetPosition()};

        position_[side] += qty;
        switch (product_type_)
        {
        case ProductType::Security:
            investment_[side] += qty * price * multiplier_;
            break;
        case ProductType::Future:
            investment_[side] += qty * price * multiplier_;
            break;
        case ProductType::Option:
            investment_[side] += qty * price * multiplier_;
            break;
        case ProductType::Perp:
            investment_[side] += qty * price * multiplier_;
            break;
        default:
            investment_[side] += qty * price;
            break;
        }
        hit_count_[side] += 1;

        const BookQty new_position{GetPosition()};
        if ((old_position < 0.0 && new_position >= 0.0) ||
            (new_position <= 0.0 && old_position > 0.0))
        {
            ++zero_cross_count_;
        }

        if (VERY_LIKELY(should_update_cost))
        {
            cost_ += GetTradeCost(side, price, qty);
        }
    }

    inline double GetTradeCost(const BookSide side, const int64_t price, const int64_t qty) const
    {
        switch (product_type_)
        {
        case ProductType::Security:
            return qty *
                   (price / GetSymbol()->GetDecimalConverter() * (fee_rate_ + tax_rate_ * (!side)) +
                    fee_cost_) *
                   multiplier_;
        case ProductType::Future:
            return qty * (price / GetSymbol()->GetDecimalConverter() * fee_rate_ + fee_cost_) *
                   multiplier_;
        case ProductType::Option:
            return qty * (price / GetSymbol()->GetDecimalConverter() * fee_rate_ + fee_cost_) *
                   multiplier_;
        case ProductType::Perp:
            return qty * (price / GetSymbol()->GetDecimalConverter() * fee_rate_ + fee_cost_) *
                   multiplier_;
        default:
            return qty * (price / GetSymbol()->GetDecimalConverter() * fee_rate_ + fee_cost_) *
                   multiplier_;
        }
    }

    inline const Symbol *GetSymbol() const
    {
        return symbol_;
    }

    inline BookQty GetPosition() const
    {
        return (position_[Bid] - position_[Ask]);
    }

    inline double GetInvestment() const
    {
        return (investment_[Bid] - investment_[Ask]) / GetSymbol()->GetDecimalConverter();
    }

    inline BookQty GetTurnOver() const
    {
        return (position_[Bid] + position_[Ask]);
    }

    inline double GetProfitOrLoss(const BookPrice &market_price) const
    {
        switch (product_type_)
        {
        case ProductType::Security:
            return GetPosition() * market_price * multiplier_ - GetInvestment();
        case ProductType::Future:
            return GetPosition() * market_price * multiplier_ - GetInvestment();
        case ProductType::Option:
            return GetPosition() * market_price * multiplier_ - GetInvestment();
        case ProductType::Perp:
            return GetPosition() * market_price * multiplier_ - GetInvestment();
        default:
            return GetPosition() * market_price * multiplier_ - GetInvestment();
        }
    }

    inline double GetProfitOrLossGrossValue(const double settle_price = NaN) const
    {
        if (std::isnan(settle_price))
        {
            if (book_ != nullptr)
            {
                if (book_->IsValid())
                {
                    return GetProfitOrLoss(book_->GetMidPrice());
                }

                if (const auto &price{book_->GetPrice(BID)}; book_->IsValidBid() && price != 0.0)
                {
                    return GetProfitOrLoss(price);
                }

                if (const auto &price{book_->GetPrice(ASK)}; book_->IsValidAsk() && price != 0.0)
                {
                    return GetProfitOrLoss(price);
                }

                return GetProfitOrLoss(book_->GetLastTradePrice());
            }
        }

        return GetProfitOrLoss(settle_price);
    }

    inline double GetProfitOrLossNetValue(const double settle_price = NaN) const
    {
        return GetProfitOrLossGrossValue(settle_price) - GetCost();
    }

    inline unsigned int GetCountZeroCrosses() const
    {
        return zero_cross_count_;
    }

    inline double GetCost() const
    {
        return cost_;
    }

    inline nlohmann::json GetJson() const
    {
        nlohmann::json json;
        json["Position"]               = GetPosition();
        json["ProfitOrLossGrossValue"] = GetProfitOrLossGrossValue();
        json["ProfitOrLossNetValue"]   = GetProfitOrLossNetValue();
        json["Cost"]                   = GetCost();
        json["TurnOver"]               = GetTurnOver();
        json["CountZeroCrosses"]       = GetCountZeroCrosses();
        return json;
    }

    inline nlohmann::json GetJson(const Symbol *s, const double m)
    {
        nlohmann::json json;
        json["Position"]               = GetPosition();
        json["ProfitOrLossGrossValue"] = GetProfitOrLossGrossValue();
        json["ProfitOrLossNetValue"]   = GetProfitOrLossNetValue();
        json["Cost"]                   = GetCost();
        json["TurnOver"]               = GetTurnOver();
        json["CountZeroCrosses"]       = GetCountZeroCrosses();
        json["Symbol"]                 = s->to_string();
        json["Multiplier"]             = m;
        return json;
    }

    const Symbol *    symbol_;
    const Book *      book_;
    const ProductType product_type_;
    const int         multiplier_;
    nlohmann::json    json_;
    const double      fee_rate_;
    const double      tax_rate_;
    const double      fee_cost_;
    int               position_[AskBid];
    int64_t           investment_[AskBid];
    unsigned int      hit_count_[AskBid];
    unsigned int      zero_cross_count_;
    double            cost_;
};

void to_json(nlohmann::json &j, const Position &p);

}  // namespace alphaone
#endif
