#include "RiskController.h"

namespace alphaone
{
RiskController::RiskController(const MultiBookManager *multi_book_manager, const std::string &name)
    : name_{name}
    , multi_book_manager_{multi_book_manager}

    , SOFT_MAX_ORDERS_PER_SEC_{40}
    , SOFT_MAX_ORDERS_PER_MIN_{10 * SOFT_MAX_ORDERS_PER_SEC_}
    , SOFT_MAX_DISTANCE_FROM_LAST_TRADE_{0.01}  // 0.01 means 1% from last trade
    , SOFT_MAX_DISTANCE_CROSS_TOUCH_{0.01}      // 0.01 means 1% worse than touch across the book
    , SOFT_MAX_BOUNCES_PER_MIN_{40}
    , SOFT_MAX_REPEATED_ORDERS_PER_MIN_{80}

    , HARD_MAX_TOTAL_ORDERS_{100000}

    , HARD_MAX_ORDERS_PER_SEC_{2 * SOFT_MAX_ORDERS_PER_SEC_}
    , HARD_MAX_ORDERS_PER_MIN_{2 * SOFT_MAX_ORDERS_PER_MIN_}
    , HARD_MAX_DISTANCE_FROM_LAST_TRADE_{2 * SOFT_MAX_DISTANCE_FROM_LAST_TRADE_}
    , HARD_MAX_DISTANCE_CROSS_TOUCH_{2 * SOFT_MAX_DISTANCE_CROSS_TOUCH_}
    , HARD_MAX_BOUNCES_PER_MIN_{2 * SOFT_MAX_BOUNCES_PER_MIN_}
    , HARD_MAX_REPEATED_ORDERS_PER_MIN_{2 * SOFT_MAX_REPEATED_ORDERS_PER_MIN_}

    , last_check_sec_{Timestamp::invalid()}
    , last_check_min_{Timestamp::invalid()}

    , soft_triggered_{Timestamp::invalid()}

    , risk_status_{RiskStatus::Good}

    , total_orders_{0}
    , orders_this_sec_{0}
    , orders_this_min_{0}
    , bounces_this_min_{0}
    , repeated_orders_this_min_{0}

    , last_0_sent_order_side_{BID}
    , last_0_sent_order_price_{0.0}
    , last_0_sent_order_qty_{0.0}
    , last_1_sent_order_side_{BID}
    , last_1_sent_order_price_{0.0}
    , last_1_sent_order_qty_{0.0}
{
    if (name == "Gaia")
    {
        const_cast<size_t &>(SOFT_MAX_ORDERS_PER_SEC_)           = 40;
        const_cast<size_t &>(SOFT_MAX_ORDERS_PER_MIN_)           = 10 * SOFT_MAX_ORDERS_PER_SEC_;
        const_cast<double &>(SOFT_MAX_DISTANCE_FROM_LAST_TRADE_) = 0.02;
        const_cast<double &>(SOFT_MAX_DISTANCE_CROSS_TOUCH_)     = 0.01;
        const_cast<size_t &>(SOFT_MAX_BOUNCES_PER_MIN_)          = 40;
        const_cast<size_t &>(SOFT_MAX_REPEATED_ORDERS_PER_MIN_)  = 80;

        const_cast<size_t &>(HARD_MAX_TOTAL_ORDERS_) = 100000;

        const_cast<size_t &>(HARD_MAX_ORDERS_PER_SEC_) = 2 * SOFT_MAX_ORDERS_PER_SEC_;
        const_cast<size_t &>(HARD_MAX_ORDERS_PER_MIN_) = 2 * SOFT_MAX_ORDERS_PER_MIN_;
        const_cast<double &>(HARD_MAX_DISTANCE_FROM_LAST_TRADE_) =
            2 * SOFT_MAX_DISTANCE_FROM_LAST_TRADE_;
        const_cast<double &>(HARD_MAX_DISTANCE_CROSS_TOUCH_) = 2 * SOFT_MAX_DISTANCE_CROSS_TOUCH_;
        const_cast<size_t &>(HARD_MAX_BOUNCES_PER_MIN_)      = 2 * SOFT_MAX_BOUNCES_PER_MIN_;
        const_cast<size_t &>(HARD_MAX_REPEATED_ORDERS_PER_MIN_) =
            2 * SOFT_MAX_REPEATED_ORDERS_PER_MIN_;
    }
    else if (name == "Zeus")
    {
        const_cast<size_t &>(SOFT_MAX_ORDERS_PER_SEC_)           = 200;
        const_cast<size_t &>(SOFT_MAX_ORDERS_PER_MIN_)           = 10 * SOFT_MAX_ORDERS_PER_SEC_;
        const_cast<double &>(SOFT_MAX_DISTANCE_FROM_LAST_TRADE_) = 0.02;
        const_cast<double &>(SOFT_MAX_DISTANCE_CROSS_TOUCH_)     = 0.01;
        const_cast<size_t &>(SOFT_MAX_BOUNCES_PER_MIN_)          = 10;
        const_cast<size_t &>(SOFT_MAX_REPEATED_ORDERS_PER_MIN_)  = 200;

        const_cast<size_t &>(HARD_MAX_TOTAL_ORDERS_) = 1000000;

        const_cast<size_t &>(HARD_MAX_ORDERS_PER_SEC_) = 2 * SOFT_MAX_ORDERS_PER_SEC_;
        const_cast<size_t &>(HARD_MAX_ORDERS_PER_MIN_) = 2 * SOFT_MAX_ORDERS_PER_MIN_;
        const_cast<double &>(HARD_MAX_DISTANCE_FROM_LAST_TRADE_) =
            2 * SOFT_MAX_DISTANCE_FROM_LAST_TRADE_;
        const_cast<double &>(HARD_MAX_DISTANCE_CROSS_TOUCH_) = 2 * SOFT_MAX_DISTANCE_CROSS_TOUCH_;
        const_cast<size_t &>(HARD_MAX_BOUNCES_PER_MIN_)      = 2 * SOFT_MAX_BOUNCES_PER_MIN_;
        const_cast<size_t &>(HARD_MAX_REPEATED_ORDERS_PER_MIN_) =
            2 * SOFT_MAX_REPEATED_ORDERS_PER_MIN_;
    }

    risk_parameters_["SOFT_MAX_ORDERS_PER_SEC"]           = SOFT_MAX_ORDERS_PER_SEC_;
    risk_parameters_["SOFT_MAX_ORDERS_PER_MIN"]           = SOFT_MAX_ORDERS_PER_MIN_;
    risk_parameters_["SOFT_MAX_DISTANCE_FROM_LAST_TRADE"] = SOFT_MAX_DISTANCE_FROM_LAST_TRADE_;
    risk_parameters_["SOFT_MAX_DISTANCE_CROSS_TOUCH"]     = SOFT_MAX_DISTANCE_CROSS_TOUCH_;
    risk_parameters_["SOFT_MAX_BOUNCES_PER_MIN"]          = SOFT_MAX_BOUNCES_PER_MIN_;
    risk_parameters_["SOFT_MAX_REPEATED_ORDERS_PER_MIN"]  = SOFT_MAX_REPEATED_ORDERS_PER_MIN_;
    risk_parameters_["HARD_MAX_TOTAL_ORDERS"]             = HARD_MAX_TOTAL_ORDERS_;
    risk_parameters_["HARD_MAX_ORDERS_PER_SEC"]           = HARD_MAX_ORDERS_PER_SEC_;
    risk_parameters_["HARD_MAX_ORDERS_PER_MIN"]           = HARD_MAX_ORDERS_PER_MIN_;
    risk_parameters_["HARD_MAX_DISTANCE_FROM_LAST_TRADE"] = HARD_MAX_DISTANCE_FROM_LAST_TRADE_;
    risk_parameters_["HARD_MAX_DISTANCE_CROSS_TOUCH"]     = HARD_MAX_DISTANCE_CROSS_TOUCH_;
    risk_parameters_["HARD_MAX_BOUNCES_PER_MIN"]          = HARD_MAX_BOUNCES_PER_MIN_;
    risk_parameters_["HARD_MAX_REPEATED_ORDERS_PER_MIN"]  = HARD_MAX_REPEATED_ORDERS_PER_MIN_;
}

const nlohmann::json &RiskController::GetRiskParameters() const
{
    return risk_parameters_;
}
}  // namespace alphaone
