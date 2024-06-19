#ifndef _FEEDPRICECHECKER_H_
#define _FEEDPRICECHECKER_H_

#include "src/infrastructure/common/typedef/Typedefs.h"

namespace alphaone
{

enum class SpotJumpStatus
{
    SoftJump,
    HardJump,
    NoJump,
};

enum class SpotJumpCheckingType
{
    No,
    Book,
    BookAndTrade,
};

class FeedPriceChecker
{
  public:
    FeedPriceChecker(double soft_jump_ratio, double hard_jump_ratio);
    ~FeedPriceChecker();

    void UpdatePrice(BookPrice mid)
    {
        last_mid_ = mid;
    }

    void UpdatePrice(BookPrice bid, BookPrice ask, BookPrice trade)
    {
        last_bid_   = bid;
        last_ask_   = ask;
        last_trade_ = trade;
    }

    SpotJumpStatus GetJumpStatus(BookPrice mid)
    {
        if (!update_count_++)
        {
            UpdatePrice(mid);
            return SpotJumpStatus::NoJump;
        }

        const auto &jump_ratio = std::fabs(last_mid_ / mid - 1.0);
        UpdatePrice(mid);
        if (jump_ratio >= hard_jump_ratio_)
            return SpotJumpStatus::HardJump;
        if (jump_ratio >= soft_jump_ratio_)
            return SpotJumpStatus::SoftJump;
        return SpotJumpStatus::NoJump;
    }

    SpotJumpStatus GetJumpStatus(BookPrice bid, BookPrice ask, BookPrice trade)
    {
        if (!update_count_++)
        {
            UpdatePrice(bid, ask, trade);
            return SpotJumpStatus::NoJump;
        }

        const auto jump_return_bid{std::abs(last_bid_ / bid - 1.0)};
        const auto jump_return_ask{std::abs(last_ask_ / ask - 1.0)};
        const auto jump_return_trade{std::abs(last_trade_ / trade - 1.0)};
        const auto jump_return{std::max({jump_return_bid, jump_return_ask, jump_return_trade})};

        UpdatePrice(bid, ask, trade);

        if (jump_return >= hard_jump_ratio_)
        {
            return SpotJumpStatus::HardJump;
        }
        if (jump_return >= soft_jump_ratio_)
        {
            return SpotJumpStatus::SoftJump;
        }
        return SpotJumpStatus::NoJump;
    }

  private:
    const double soft_jump_ratio_;
    const double hard_jump_ratio_;

    BookPrice last_mid_;
    BookPrice last_bid_;
    BookPrice last_ask_;
    BookPrice last_trade_;
    size_t    update_count_;
};


}  // namespace alphaone

#endif
