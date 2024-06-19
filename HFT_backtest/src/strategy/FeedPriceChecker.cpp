#include "FeedPriceChecker.h"

namespace alphaone
{

FeedPriceChecker::FeedPriceChecker(double soft_jump_ratio, double hard_jump_ratio)
    : soft_jump_ratio_{soft_jump_ratio}
    , hard_jump_ratio_{hard_jump_ratio}
    , last_mid_{std::nan("")}
    , update_count_{0}
{
    if (soft_jump_ratio_ >= hard_jump_ratio_)
        throw std::invalid_argument(
            fmt::format("soft jump ratio {:.2f} cannot be bigger than hard jump ratio {:.2f}",
                        soft_jump_ratio_, hard_jump_ratio_));
}

FeedPriceChecker::~FeedPriceChecker()
{
}


}  // namespace alphaone
