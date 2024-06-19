#include "Position.h"

namespace alphaone
{

void to_json(nlohmann::json &j, const Position &p)
{
    j = nlohmann::json{{"symbol", p.symbol_->to_string()},
                       {"last_event_loop_time", p.book_->GetLastEventLoopTime().to_string()},
                       {"product_type", EnumToString::ToString(p.product_type_)},
                       {"multiplier", p.multiplier_},
                       {"fee_rate", p.fee_rate_},
                       {"fee_cost", p.fee_cost_},
                       {"position_bid", p.position_[Bid]},
                       {"position_ask", p.position_[Ask]},
                       {"investment_bid", p.investment_[Bid]},
                       {"investment_ask", p.investment_[Ask]},
                       {"hit_count_bid", p.hit_count_[Bid]},
                       {"hit_count_ask", p.hit_count_[Ask]},
                       {"zero_cross_count", p.zero_cross_count_},
                       {"cost", p.cost_}};
}

}  // namespace alphaone
