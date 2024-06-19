#include "TaifexSimulator.h"

namespace alphaone
{
TaifexSimulator::TaifexSimulator(const GlobalConfiguration *config, const Symbol *symbol,
                                 const MultiBookManager *multi_book_manager, Engine *engine,
                                 TaifexSimulationOrderManager *order_manager)
    : config_{config}
    , json_(config_->GetJson().at("Simulation"))
    , symbol_{symbol}
    , mode_{SimulationMode::ImmediatelyFilled}
    , multi_book_manager_{multi_book_manager}
    , pm_{static_cast<int>(symbol_->GetDecimalConverter())}
    , timer_mode_{TimerMode::PacketEnd}
    , market_impact_reset_mode_{MarketImpactResetMode::SingleTickReset}
    , forward_delay_{Duration::from_sec(json_.value("passive_forward_delay_avg", 0.)),
                     Duration::from_sec(json_.value("aggressive_forward_delay_avg", 0.))}
    , backward_delay_{Duration::from_sec(json_.value("passive_backward_delay_avg", 0.)),
                      Duration::from_sec(json_.value("aggressive_backward_delay_avg", 0.))}
    , rod_report_delay_{Duration::from_sec(json_.value("rod_report_delay_avg", 0.))}
    , cancel_delay_{json_.contains("cancel_delay_avg")
                        ? Duration::from_sec(json_["cancel_delay_avg"].get<double>())
                        : forward_delay_[Passive]}
    , cancel_report_delay_{std::max(cancel_delay_, forward_delay_[Aggressive])}
    , double_order_delay_{Duration::from_sec(json_.value("double_order_delay_avg", 2.5 * 1e-6))}
    , peek_num_of_levels_{json_.value("peek_number_of_levels", 5)}
    , match_time_{Duration::from_sec(json_.value("match_time", 1e-7))}
    , is_optimistic_when_filled_by_bookdiff_{false}
    , is_optimistic_of_queue_position_{json_.value("is_optimistic_of_queue_position", false)}
    , book_{dynamic_cast<MarketByOrderBook *>(multi_book_manager_->GetBook(symbol_))}
    , numbers_of_trade_to_check_{0}
    , possible_trades_window_{Duration::from_sec(json_.value("possible_trades_window", 3.))}
    , engine_{engine}
    , order_manager_{order_manager}
    , is_considering_implied_{json_.value("is_considering_implied", false)}
    , double_order_percentage_{json_.value("double_order_percentage", 0.15)}
    , double_order_qty_{json_.value("double_order_qty", 20)}
    , qty_in_front_{json_.value("qty_in_front_ask", 0), json_.value("qty_in_front_bid", 0)}
    , order_shift_{Duration::from_sec(json_.value("order_shift", 1e-7))}
    , touch_price_{INT32_MIN, INT32_MAX}
    , double_order_no_{-1}
    , calc_time_{json_.value("calc_time", 1e-4)}
    , curr_seqno_{0}
    , delay_root_path_{json_.value("delay_root_path", "/var/files/Delay/Day")}
    , delay_file_name_{json_.value("delay_file_name", "delay.csv.gz")}
    , is_force_passive_{json_.value("is_force_passive", false)}
{
    Init();

    qty_taken_from_book_total_[Ask] =
        std::map<int, int, dynamic_compare>(dynamic_compare(dynamic_compare::less));
    qty_taken_from_book_total_[Bid] =
        std::map<int, int, dynamic_compare>(dynamic_compare(dynamic_compare::greater));

    order_queue_position_[Ask] =
        std::map<int, std::map<const TaifexOnExchOrder *, QueuePosition>, dynamic_compare>(
            dynamic_compare(dynamic_compare::less));
    order_queue_position_[Bid] =
        std::map<int, std::map<const TaifexOnExchOrder *, QueuePosition>, dynamic_compare>(
            dynamic_compare(dynamic_compare::greater));

    order_reports_pool_ = new boost::pool<boost::default_user_allocator_malloc_free>(
        sizeof(TaifexSimulationOrderReportMessage), 256);
    on_exch_orders_pool_ =
        new boost::pool<boost::default_user_allocator_malloc_free>(sizeof(TaifexOnExchOrder), 256);
    // this book should be cached as one
    prev_book_pool_ =
        new boost::pool<boost::default_user_allocator_malloc_free>(sizeof(BookCache), 128);
    cancel_info_pool_ =
        new boost::pool<boost::default_user_allocator_malloc_free>(sizeof(CancelOrderData), 128);
    order_reports_.reserve(128);

    if (json_.contains("DelayReader"))
    {
        auto &delay_json = json_["DelayReader"];
        if (!delay_json.contains("file_path"))
        {
            delay_json["file_path"] =
                fmt::format("{}/{}/{}", delay_root_path_, engine_->GetDate().to_string_with_dash(),
                            delay_file_name_);
        }
        delay_reader_ = new DelayReader(json_["DelayReader"]);
    }
}

TaifexSimulator::~TaifexSimulator()
{
    for (auto &[order, book_cache] : order_to_prev_book_cache_map_)
    {
        delete (*book_cache)[Ask];
        delete (*book_cache)[Bid];
    }

    if (delay_reader_)
    {
        delete delay_reader_;
        delay_reader_ = nullptr;
    }
    if (order_reports_pool_)
    {
        delete order_reports_pool_;
        order_reports_pool_ = nullptr;
    }
    if (on_exch_orders_pool_)
    {
        delete on_exch_orders_pool_;
        on_exch_orders_pool_ = nullptr;
    }
    if (prev_book_pool_)
    {
        delete prev_book_pool_;
        prev_book_pool_ = nullptr;
    }
    if (cancel_info_pool_)
    {
        delete cancel_info_pool_;
        cancel_info_pool_ = nullptr;
    }
}

void TaifexSimulator::Init()
{
    // construct simulation_mode_string_to_enum_map_
    for (int i = SimulationMode::ImmediatelyFilled; i < SimulationMode::LastSimulationMode; ++i)
    {
        auto mode = static_cast<SimulationMode>(i);
        simulation_mode_string_to_enum_map_[SimulationModeToString(mode)] = mode;
    }

    for (int i = TimerMode::PacketEnd; i < TimerMode::LastTimerMode; ++i)
    {
        auto t_mode                                               = static_cast<TimerMode>(i);
        timer_mode_string_to_enum_map_[TimerModeToString(t_mode)] = t_mode;
    }

    for (int i = MarketImpactResetMode::Off; i < MarketImpactResetMode::LastResetMode; ++i)
    {
        auto m_mode = static_cast<MarketImpactResetMode>(i);
        market_reset_to_enum_map_[MarketImpactResetModeToString(m_mode)] = m_mode;
    }

    const auto &mode = json_["simulation_mode"];
    if (mode.is_number_integer())
    {
        mode_ = static_cast<SimulationMode>(mode.get<int>());
        SPDLOG_INFO("[{}] [{}] Set simulation_mode to {}", __func__, symbol_->to_string(),
                    SimulationModeToString(mode_));
    }
    else if (mode.is_string())
    {
        if (auto it = simulation_mode_string_to_enum_map_.find(mode.get<std::string>());
            it != simulation_mode_string_to_enum_map_.end())
        {
            mode_ = it->second;
            SPDLOG_INFO("[{}] [{}] Set simulation_mode to {}", __func__, symbol_->to_string(),
                        it->first);
        }
    }

    const auto &t_mode = json_["timer_mode"];
    if (t_mode.is_number_integer())
    {
        timer_mode_ = static_cast<TimerMode>(t_mode.get<int>());
        SPDLOG_INFO("[{}] [{}] Set timer_mode to {}", __func__, symbol_->to_string(),
                    TimerModeToString(timer_mode_));
    }
    else if (t_mode.is_string())
    {
        if (auto it = timer_mode_string_to_enum_map_.find(t_mode.get<std::string>());
            it != timer_mode_string_to_enum_map_.end())
        {
            timer_mode_ = it->second;
            SPDLOG_INFO("[{}] [{}] Set timer_mode to {}", __func__, symbol_->to_string(),
                        it->first);
        }
    }

    const auto &m_mode = json_["market_impact_reset_mode"];
    if (m_mode.is_number_integer())
    {
        market_impact_reset_mode_ = static_cast<MarketImpactResetMode>(m_mode.get<int>());
        SPDLOG_INFO("[{}] [{}] Set timer_mode to {}", __func__, symbol_->to_string(),
                    MarketImpactResetModeToString(market_impact_reset_mode_));
    }
    else if (m_mode.is_string())
    {
        if (auto it = market_reset_to_enum_map_.find(m_mode.get<std::string>());
            it != market_reset_to_enum_map_.end())
        {
            market_impact_reset_mode_ = it->second;
            SPDLOG_INFO("[{}] [{}] Set timer_mode to {}", __func__, symbol_->to_string(),
                        it->first);
        }
    }


    if (mode_ == SimulationMode::FilledByBookDiff)
    {
        if (json_.at("is_optimistic_when_filled_by_bookdiff").is_boolean())
        {
            is_optimistic_when_filled_by_bookdiff_ =
                json_["is_optimistic_when_filled_by_bookdiff"].get<bool>();
            SPDLOG_INFO("[{}] [{}] is_optimistic_when_filled_by_bookdiff_ = {}", __func__,
                        symbol_->to_string(), is_optimistic_when_filled_by_bookdiff_);
        }
    }
}

std::array<int, AskBid> TaifexSimulator::GetTouchPair(BookSide side, BookPrice price,
                                                      BookPrice implied_bid, BookPrice implied_ask)
{
    std::array<int, AskBid> bid_ask_on_book_size{0, 0};
    bid_ask_on_book_size[side] = static_cast<int>(book_->GetQtyBehindPrice(side, price));
    if (!is_considering_implied_)
        return bid_ask_on_book_size;
    if (side == BID && implied_bid >= price)
        bid_ask_on_book_size[side] += static_cast<int>(book_->GetImpliedOrderQty(Bid));
    else if (side == ASK && implied_ask <= price)
        bid_ask_on_book_size[side] += static_cast<int>(book_->GetImpliedOrderQty(Ask));
    return bid_ask_on_book_size;
}

void TaifexSimulator::ProcessNewOrder(int orderno, int price, int qty,
                                      const TAIFEX_ORDER_SIDE &          side,
                                      const TAIFEX_ORDER_TIMEINFORCE &   timeInForce,
                                      const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                                      const Timestamp &                  event_loop_time)
{
    auto        bid         = book_->GetTouchLevel(BidType::GetType())->price_;
    auto        ask         = book_->GetTouchLevel(AskType::GetType())->price_;
    const auto &implied_bid = book_->GetImpliedOrderPrice(Bid);
    const auto &implied_ask = book_->GetImpliedOrderPrice(Ask);

    if (is_considering_implied_)
    {
        bid = std::max(bid, implied_bid);
        ask = std::min(ask, implied_ask);
    }

    touch_price_[Bid] = static_cast<int>(std::nearbyint(bid * pm_));
    touch_price_[Ask] = static_cast<int>(std::nearbyint(ask * pm_));

    const auto  is_bid{side == TAIFEX_ORDER_SIDE::BUY};
    const auto  is_aggresive{is_bid ? (price >= touch_price_[Ask] && !is_force_passive_)
                                    : (price <= touch_price_[Bid] && !is_force_passive_)};
    const auto &exch_received_time = event_loop_time + forward_delay_[is_aggresive];
    // deal with new orders by different
    switch (mode_)
    {
    case SimulationMode::ImmediatelyFilled:
    {
        switch (timeInForce)
        {
        case TAIFEX_ORDER_TIMEINFORCE::IOC:
        {
            // find how much could be filled
            auto exec_pairs = is_bid ? AggressiveMatch<BidType, false>(price, qty, timeInForce)
                                     : AggressiveMatch<AskType, false>(price, qty, timeInForce);
            InsertIocReport(event_loop_time, orderno, price, qty, side, timeInForce, positionEffect,
                            &exec_pairs);
            break;
        }
        case TAIFEX_ORDER_TIMEINFORCE::FOK:
        {
            // find how much could be filled
            auto exec_pairs = is_bid ? AggressiveMatch<BidType, false>(price, qty, timeInForce)
                                     : AggressiveMatch<AskType, false>(price, qty, timeInForce);
            InsertFokReport(event_loop_time, orderno, price, qty, side, timeInForce, positionEffect,
                            &exec_pairs);
            break;
        }
        case TAIFEX_ORDER_TIMEINFORCE::ROD:
        {
            auto remaining_qty{qty};
            auto last_executed_ts{Timestamp::invalid()};
            // aggressive ROD order
            if (is_aggresive)
            {
                auto exec_pairs  = is_bid ? AggressiveMatch<BidType, false>(price, qty, timeInForce)
                                          : AggressiveMatch<AskType, false>(price, qty, timeInForce);
                remaining_qty    = InsertRodReport(event_loop_time, orderno, price, qty, side,
                                                timeInForce, positionEffect, &exec_pairs);
                last_executed_ts = exch_received_time;
            }

            // passive ROD order and agreesive with extra qty
            if (remaining_qty > 0)
            {
                auto *oeo = GetOnExchOrderFromPool(price, qty, remaining_qty, orderno, side,
                                                   event_loop_time, exch_received_time,
                                                   last_executed_ts, timeInForce, positionEffect);
                // add a new outstanding order for the remaining size
                order_no_to_on_exch_orders_map_[orderno] = oeo;
                passive_orders_queue_.push_back(oeo);
                // insert with timer prior to match
                engine_->AddOneTimeTimer(
                    event_loop_time + order_shift_,
                    &alphaone::TaifexSimulator::InsertOrderQueuePositionCallBack, this, oeo);
            }

            break;
        }
        default:
            break;
        }

        break;
    }
    case SimulationMode::FilledByFillModel:
        break;
    case SimulationMode::FilledByBookDiffAndTrade:
    case SimulationMode::FilledByBookDiff:
    {
        auto *      oeo = GetOnExchOrderFromPool(price, qty, qty, orderno, side, event_loop_time,
                                           exch_received_time, Timestamp::invalid(), timeInForce,
                                           positionEffect);
        auto        need_process = false;
        const auto &adj_price    = price / (double)pm_;
        switch (timeInForce)
        {
        case TAIFEX_ORDER_TIMEINFORCE::IOC:
        case TAIFEX_ORDER_TIMEINFORCE::FOK:
        {
            aggressive_order_size_pairs_.push(
                {oeo, GetTouchPair(!is_bid, adj_price, implied_bid, implied_ask)});
            need_process = true;
            break;
        }
        case TAIFEX_ORDER_TIMEINFORCE::ROD:
        {
            // aggressive check the other side
            if (is_aggresive)
            {
                aggressive_order_size_pairs_.push(
                    {oeo, GetTouchPair(!is_bid, adj_price, implied_bid, implied_ask)});
                need_process = true;
            }
            else
            // passive check the same side for target price
            {
                const bool &check_side = is_bid;
                auto        book_cache = GetBookCacheFromPool();
                InitBookCache(book_cache);
                RefreshBookCache(book_cache, check_side);
                order_to_prev_book_cache_map_[oeo]       = book_cache;
                order_no_to_on_exch_orders_map_[orderno] = oeo;
                passive_orders_queue_.push_back(oeo);
                // insert with timer prior to match
                engine_->AddOneTimeTimer(
                    event_loop_time + order_shift_,
                    &alphaone::TaifexSimulator::InsertOrderQueuePositionCallBack, this, oeo);
            }

            break;
        }
        default:
            break;
        }

        if (timer_mode_ == TimerMode::AccurateTimer || need_process)
        {
            engine_->AddOneTimeTimer(exch_received_time,
                                     &TaifexSimulator::ProcessOrderQueueCallBack, this, nullptr);
        }

        break;
    }
    default:
        break;
    }
}

void TaifexSimulator::NewOrder(const int orderno, const char *pid, int price, int qty,
                               TAIFEX_ORDER_SIDE side, TAIFEX_ORDER_TIMEINFORCE timeInForce,
                               TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                               const Timestamp &           received_event_loop_time)
{
    if (delay_reader_)
        forward_delay_[Aggressive] =
            Duration::from_sec(delay_reader_->GetDelay(curr_seqno_, calc_time_));

    const auto &received_time =
        received_event_loop_time + forward_delay_[Passive] + backward_delay_[Passive];
    if (BRANCH_UNLIKELY(!book_->IsValid() || qty <= 0))
    {
        InsertOrderReportWithTimer(price, 0, qty, side, timeInForce, positionEffect,
                                   OrderReportType::OrderRejected, orderno, received_time);
        return;
    }
    else
    {
        InsertOrderReportWithTimer(price, qty, 0, side, timeInForce, positionEffect,
                                   OrderReportType::OrderAccepted, orderno, received_time);
    }

    ProcessNewOrder(orderno, price, qty, side, timeInForce, positionEffect,
                    received_event_loop_time);
}

void TaifexSimulator::CancelOrder(const int orderno, const char *pid, TAIFEX_ORDER_SIDE side,
                                  bool             is_need_cancel_fail,
                                  const Timestamp &received_event_loop_time)
{
    if (delay_reader_)
        forward_delay_[Aggressive] =
            Duration::from_sec(delay_reader_->GetDelay(curr_seqno_, calc_time_));

    const auto &processed_cancel_time = received_event_loop_time + cancel_delay_;
    const auto &processed_report_time = received_event_loop_time + cancel_report_delay_;

    auto cancel_order_data                  = GetCancelOrderDataFromPool();
    cancel_order_data->order_no_            = orderno;
    cancel_order_data->pid_                 = pid;
    cancel_order_data->side_                = side;
    cancel_order_data->is_need_cancel_fail_ = is_need_cancel_fail;
    cancel_order_data->cancel_sent_ts_      = received_event_loop_time;

    engine_->AddOneTimeTimer(processed_cancel_time,
                             &TaifexSimulator::CleanPassiveOrderResourcesCallBack, this,
                             cancel_order_data);
    engine_->AddOneTimeTimer(processed_report_time, &TaifexSimulator::ProcessOrderQueueCallBack,
                             this, nullptr);
}

void TaifexSimulator::NewDoubleOrder(const int orderno, const char *pid, int bidprice, int bidqty,
                                     int askprice, int askqty, TAIFEX_ORDER_TIMEINFORCE timeInForce,
                                     TAIFEX_ORDER_POSITIONEFFECT positionEffect,
                                     const Timestamp &           received_event_loop_time)
{
    if (delay_reader_)
        forward_delay_[Aggressive] =
            Duration::from_sec(delay_reader_->GetDelay(curr_seqno_, calc_time_));

    const auto &received_time =
        received_event_loop_time + forward_delay_[Passive] + backward_delay_[Passive];
    auto invalid_quote = bidprice * (1 + double_order_percentage_) < askprice ||
                         bidprice >= askprice ||
                         positionEffect != TAIFEX_ORDER_POSITIONEFFECT::QUOTE ||
                         bidqty < double_order_qty_ || askqty < double_order_qty_;
    if (BRANCH_UNLIKELY(!book_->IsValid() || invalid_quote))
    {
        InsertOrderReportWithTimer(bidprice, 0, bidqty, TAIFEX_ORDER_SIDE::BUY, timeInForce,
                                   positionEffect, OrderReportType::OrderRejected, orderno,
                                   received_time);
        InsertOrderReportWithTimer(askprice, 0, askqty, TAIFEX_ORDER_SIDE::SELL, timeInForce,
                                   positionEffect, OrderReportType::OrderRejected, orderno,
                                   received_time);
        return;
    }

    if (double_order_no_ >= 0)
    {
        CancelOrder(double_order_no_, pid, TAIFEX_ORDER_SIDE::BUY, false, received_event_loop_time);
        CancelOrder(-double_order_no_, pid, TAIFEX_ORDER_SIDE::SELL, false,
                    received_event_loop_time);
    }
    const auto &double_received_time = received_time + double_order_delay_;
    InsertOrderReportWithTimer(bidprice, bidqty, 0, TAIFEX_ORDER_SIDE::BUY, timeInForce,
                               positionEffect, OrderReportType::OrderAccepted, orderno,
                               double_received_time);
    InsertOrderReportWithTimer(askprice, askqty, 0, TAIFEX_ORDER_SIDE::SELL, timeInForce,
                               positionEffect, OrderReportType::OrderAccepted, orderno,
                               double_received_time);

    ProcessNewOrder(orderno, bidprice, bidqty, TAIFEX_ORDER_SIDE::BUY, timeInForce, positionEffect,
                    received_event_loop_time + double_order_delay_);
    ProcessNewOrder(-orderno, askprice, askqty, TAIFEX_ORDER_SIDE::SELL, timeInForce,
                    positionEffect, received_event_loop_time + double_order_delay_);

    double_order_no_ = orderno;
}

void TaifexSimulator::CancelDoubleOrder(const char *orderno, const char *pid,
                                        const Timestamp &received_event_loop_time)
{
    if (delay_reader_)
        forward_delay_[Aggressive] =
            Duration::from_sec(delay_reader_->GetDelay(curr_seqno_, calc_time_));

    const int   int_order_no         = OrderNoToInt(orderno);
    const auto &double_received_time = received_event_loop_time + double_order_delay_ + match_time_;
    CancelOrder(int_order_no, pid, TAIFEX_ORDER_SIDE::BUY, true, double_received_time);
    CancelOrder(-int_order_no, pid, TAIFEX_ORDER_SIDE::SELL, true, double_received_time);

    // may need to check if this is ok
    double_order_no_ = -1;
}

void TaifexSimulator::ModifyOrder(const char *orderno, const char *pid, int bidprice, int askprice,
                                  const Timestamp &received_event_loop_time)
{
    if (delay_reader_)
        forward_delay_[Aggressive] =
            Duration::from_sec(delay_reader_->GetDelay(curr_seqno_, calc_time_));

    auto        in_orderno = OrderNoToInt(orderno);
    auto        new_orderno{in_orderno};
    const auto &processed_time =
        received_event_loop_time + forward_delay_[Passive] + backward_delay_[Passive];
    auto is_cancelled{false};
    auto price{0}, qty{0};
    auto side{TAIFEX_ORDER_SIDE::BUY};
    auto bidqty{0}, askqty{0};
    if (auto ait = order_no_to_on_exch_orders_map_.find(-in_orderno);
        ait != order_no_to_on_exch_orders_map_.end())
        askqty = ait->second->remaining_qty_;
    if (auto bit = order_no_to_on_exch_orders_map_.find(in_orderno);
        bit != order_no_to_on_exch_orders_map_.end())
        bidqty = bit->second->remaining_qty_;
    if (in_orderno != double_order_no_ || (bidprice && askprice) ||
        (bidprice && bidqty < double_order_qty_) || (askprice && askqty < double_order_qty_))
    {
        InsertOrderReportWithTimer(0, 0, 0, TAIFEX_ORDER_SIDE::SELL, TAIFEX_ORDER_TIMEINFORCE::ROD,
                                   TAIFEX_ORDER_POSITIONEFFECT::QUOTE,
                                   OrderReportType::OrderModifyFailed, in_orderno, processed_time);
        return;
    }
    if (bidprice)
    {
        price = bidprice;
        std::tie(is_cancelled, qty) =
            CleanPassiveOrderResources<false>(in_orderno, received_event_loop_time, BID);
        side = TAIFEX_ORDER_SIDE::BUY;
    }
    if (askprice)
    {
        price = askprice;
        std::tie(is_cancelled, qty) =
            CleanPassiveOrderResources<false>(-in_orderno, received_event_loop_time, ASK);
        side        = TAIFEX_ORDER_SIDE::SELL;
        new_orderno = -in_orderno;
    }

    if (!is_cancelled || qty <= 0)
    {
        InsertOrderReportWithTimer(0, 0, 0, TAIFEX_ORDER_SIDE::SELL, TAIFEX_ORDER_TIMEINFORCE::ROD,
                                   TAIFEX_ORDER_POSITIONEFFECT::QUOTE,
                                   OrderReportType::OrderModifyFailed, in_orderno, processed_time);
        return;
    }

    InsertOrderReportWithTimer(price, qty, 0, side, TAIFEX_ORDER_TIMEINFORCE::ROD,
                               TAIFEX_ORDER_POSITIONEFFECT::QUOTE, OrderReportType::OrderModified,
                               in_orderno, processed_time);

    ProcessNewOrder(new_orderno, price, qty, side, TAIFEX_ORDER_TIMEINFORCE::ROD,
                    TAIFEX_ORDER_POSITIONEFFECT::QUOTE,
                    received_event_loop_time + double_order_delay_);
}

void TaifexSimulator::InsertOrderQueuePosition(TaifexOnExchOrder *order)
{
    const auto &price      = order->price_;
    const auto &side       = order->side_ == TAIFEX_ORDER_SIDE::BUY;
    const auto &book_price = price / static_cast<double>(pm_);
    auto        level      = book_->GetLevelFromPrice(side, book_price);
    const auto &implied_qty =
        is_considering_implied_ && book_->GetImpliedOrderPrice(side) == book_price
            ? book_->GetImpliedOrderQty(side)
            : 0.;
    const auto &qty =
        static_cast<int>(std::nearbyint(level ? level->qty_ + implied_qty : implied_qty));
    auto it = order_queue_position_[side].find(price);
    if (it == order_queue_position_[side].end())
    {
        it = order_queue_position_[side]
                 .insert({price, std::map<const TaifexOnExchOrder *, QueuePosition>()})
                 .first;
    }
    it->second.insert({order, {qty + qty_in_front_[side], -order->original_qty_}});
}

void TaifexSimulator::InsertOrderQueuePositionCallBack(const Timestamp event_loop_time,
                                                       const Timestamp call_back_time,
                                                       void *          structure)
{
    auto o = reinterpret_cast<TaifexOnExchOrder *>(structure);
    InsertOrderQueuePosition(o);
}

void TaifexSimulator::CleanOrderQueuePosition(const bool &side, const TaifexOnExchOrder *o)
{
    if (auto oit = order_queue_position_[side].find(o->price_);
        oit != order_queue_position_[side].end())
    {
        if (auto pit = oit->second.find(o); pit != oit->second.end())
        {
            oit->second.erase(pit);
        }
    }
}

std::pair<int, int> TaifexSimulator::GetOrderQueuePosition(const bool &             side,
                                                           const TaifexOnExchOrder *o)
{
    auto oit = order_queue_position_[side].find(o->price_);
    if (oit == order_queue_position_[side].end())
    {
        return {-1, 0};
    }

    if (auto pit = oit->second.find(o); pit != oit->second.end())
    {
        return {pit->second.position_[Before], pit->second.last_position_[Before]};
    }

    return {-1, 0};
}

int TaifexSimulator::GetQtyTakenOnBook(const BookSide &side, int exec_price)
{
    if (auto it = qty_taken_from_book_total_[side].find(exec_price);
        it != qty_taken_from_book_total_[side].end())
    {
        return it->second;
    }
    return 0;
}

void TaifexSimulator::UpdateQtyTakenOnBook(const BookSide &side, int exec_price, int exec_qty)
{
    // add how many we took, if insert fail then add it to current qty
    if (market_impact_reset_mode_ == MarketImpactResetMode::NoImpact)
    {
        return;
    }

    if (auto [it, success] =
            qty_taken_from_book_total_[side].insert({exec_price, std::max(exec_qty, 0)});
        !success)
    {
        it->second = std::max(it->second + exec_qty, 0);
    }
}

void TaifexSimulator::InitBookCache(BookCache *book_cache)
{
    for (int i = Ask; i < AskBid; ++i)
    {
        (*book_cache)[i] = new std::vector<std::pair<int, int>>(peek_num_of_levels_, {0, 0});
    }
}

void TaifexSimulator::RefreshBookCache(BookCache *book_cache, const bool &side)
{
    auto level = book_->GetTouchLevel(side);
    for (int i = 0; i < peek_num_of_levels_; ++i)
    {
        if (!level)
            break;

        (*(*book_cache)[side])[i].first  = std::nearbyint(level->price_ * pm_);
        (*(*book_cache)[side])[i].second = std::nearbyint(level->qty_);
        if (is_considering_implied_ && book_->GetImpliedOrderPrice(side) == level->price_)
            (*(*book_cache)[side])[i].second += std::nearbyint(book_->GetImpliedOrderQty(side));
        level = level->next_;
    }
}

void TaifexSimulator::CleanBookCache(BookCache *book_cache)
{
    for (int i = Ask; i < AskBid; ++i)
    {
        delete (*book_cache)[i];
    }
}

void TaifexSimulator::InsertOrderReportCallBack(const Timestamp event_loop_time,
                                                const Timestamp call_back_time, void *structure)
{
    // [TODO] maybe we can directly Process here, may need to discuss
    auto o = reinterpret_cast<TaifexSimulationOrderReportMessage *>(structure);
    order_reports_.push_back(o);
    engine_->AddOneTimeTimer(event_loop_time,
                             &alphaone::TaifexSimulationOrderManager::ProcessCallBack,
                             order_manager_, nullptr);
}

void TaifexSimulator::CleanPassiveOrderResourcesCallBack(const Timestamp event_loop_time,
                                                         const Timestamp call_back_time,
                                                         void *          structure)
{
    auto cancel_order_data = reinterpret_cast<CancelOrderData *>(structure);
    if (!CleanPassiveOrderResources<true>(cancel_order_data->order_no_,
                                          cancel_order_data->cancel_sent_ts_,
                                          cancel_order_data->side_ == TAIFEX_ORDER_SIDE::BUY)
             .first &&
        cancel_order_data->is_need_cancel_fail_)
    {
        // set order cancel failed later
        InsertOrderReportWithTimer(0, 0, 0, TAIFEX_ORDER_SIDE::BUY, TAIFEX_ORDER_TIMEINFORCE::ROD,
                                   TAIFEX_ORDER_POSITIONEFFECT::CLOSE,
                                   OrderReportType::OrderCancelFailed, cancel_order_data->order_no_,
                                   event_loop_time + backward_delay_[Passive]);
    }
    DestroyObjFromPool(cancel_order_data, cancel_info_pool_);
}

void TaifexSimulator::InsertIocReport(const Timestamp &event_loop_time, int orderno, int price,
                                      int qty, const TAIFEX_ORDER_SIDE &side,
                                      const TAIFEX_ORDER_TIMEINFORCE &   timeInForce,
                                      const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                                      std::vector<std::pair<int, int>> * exec_pairs)
{
    if (exec_pairs->size() == 0)
    {
        // no executed thus return qty = 0 and leaves qty = 0
        InsertOrderReport(0, 0, 0, side, timeInForce, positionEffect,
                          OrderReportType::OrderExecuted, orderno, event_loop_time);
        return;
    }
    auto leaves_qty{qty};
    for (auto it = exec_pairs->begin(); it != exec_pairs->end(); ++it)
    {
        // insert remaining order report, last one will set leaves qty = 0
        leaves_qty -= it->second;
        InsertOrderReport(it->first, it->second,
                          (std::next(it) == exec_pairs->end()) ? 0 : leaves_qty, side, timeInForce,
                          positionEffect, OrderReportType::OrderExecuted, orderno, event_loop_time);
    }
}

void TaifexSimulator::InsertFokReport(const Timestamp &event_loop_time, int orderno, int price,
                                      int qty, const TAIFEX_ORDER_SIDE &side,
                                      const TAIFEX_ORDER_TIMEINFORCE &   timeInForce,
                                      const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                                      std::vector<std::pair<int, int>> * exec_pairs)
{
    if (exec_pairs->size() == 0)
    {
        InsertOrderReport(0, 0, 0, side, timeInForce, positionEffect,
                          OrderReportType::OrderExecuted, orderno, event_loop_time);
        return;
    }
    auto leaves_qty{qty};
    for (const auto &pair : *exec_pairs)
    {
        leaves_qty -= pair.second;
        InsertOrderReport(pair.first, pair.second, leaves_qty, side, timeInForce, positionEffect,
                          OrderReportType::OrderExecuted, orderno, event_loop_time);
    }
}

int TaifexSimulator::InsertRodReport(const Timestamp &event_loop_time, int orderno, int price,
                                     int qty, const TAIFEX_ORDER_SIDE &side,
                                     const TAIFEX_ORDER_TIMEINFORCE &   timeInForce,
                                     const TAIFEX_ORDER_POSITIONEFFECT &positionEffect,
                                     std::vector<std::pair<int, int>> * exec_pairs)
{
    auto leaves_qty{qty};
    for (const auto &pair : *exec_pairs)
    {
        leaves_qty -= pair.second;
        InsertOrderReportWithTimer(pair.first, pair.second, leaves_qty, side, timeInForce,
                                   positionEffect, OrderReportType::OrderExecuted, orderno,
                                   event_loop_time);
    }
    return leaves_qty;
}

void TaifexSimulator::ProcessAggressiveOrderQueue(const Timestamp &event_loop_time)
{
    const auto &backward_delay = match_time_ + backward_delay_[Aggressive];
    while (!aggressive_order_size_pairs_.empty())
    {
        auto it                             = aggressive_order_size_pairs_.front();
        auto &[order, on_book_size_by_side] = it;

        if (order->order_received_by_exch_ts_ > event_loop_time)
        {
            break;
        }
        const auto &order_price       = order->price_ / (double)pm_;
        const auto &side              = order->side_ == TAIFEX_ORDER_SIDE::BUY;
        const auto &other_side        = !side;
        const auto &prev_on_book_size = on_book_size_by_side[other_side];
        const auto &curr_implied =
            is_considering_implied_ && book_->GetImpliedOrderPrice(other_side) == order_price
                ? book_->GetImpliedOrderQty(other_side)
                : 0.;

        const auto &curr_on_book_size =
            static_cast<int>(book_->GetQtyBehindPrice(other_side, order_price) + curr_implied);
        const auto final_book_size = (is_optimistic_when_filled_by_bookdiff_)
                                         ? std::max(curr_on_book_size, prev_on_book_size)
                                         : std::min(curr_on_book_size, prev_on_book_size);

        const auto max_exec_qty = std::min(final_book_size, order->remaining_qty_);
        auto       exec_pairs   = side ? GetExecPairsFromPrice<BidType>(order_price, max_exec_qty)
                                       : GetExecPairsFromPrice<AskType>(order_price, max_exec_qty);
        int        total_exec_qty{0};
        for (const auto &[price, qty] : exec_pairs)
            total_exec_qty += qty;
        const auto &remaining_qty = order->remaining_qty_ - total_exec_qty;
        switch (order->timeinforce_)
        {
        case TAIFEX_ORDER_TIMEINFORCE::IOC:
        {
            InsertIocReport(event_loop_time + backward_delay, order->order_no_, order->price_,
                            order->remaining_qty_, order->side_, order->timeinforce_,
                            order->position_effect_, &exec_pairs);

            DestroyObjFromPool<TaifexOnExchOrder>(order, on_exch_orders_pool_);

            break;
        }
        case TAIFEX_ORDER_TIMEINFORCE::FOK:
        {
            if (remaining_qty != 0)
            {
                for (const auto &[price, qty] : exec_pairs)
                    UpdateQtyTakenOnBook(other_side, price, -qty);
                exec_pairs.clear();
            }

            InsertFokReport(event_loop_time + backward_delay, order->order_no_, order->price_,
                            order->remaining_qty_, order->side_, order->timeinforce_,
                            order->position_effect_, &exec_pairs);
            DestroyObjFromPool<TaifexOnExchOrder>(order, on_exch_orders_pool_);

            break;
        }
        case TAIFEX_ORDER_TIMEINFORCE::ROD:
        {
            InsertRodReport(event_loop_time + backward_delay, order->order_no_, order->price_,
                            order->remaining_qty_, order->side_, order->timeinforce_,
                            order->position_effect_, &exec_pairs);
            if (remaining_qty > 0)
            {
                // need to check if the logic is ok
                order->remaining_qty_                             = remaining_qty;
                order_no_to_on_exch_orders_map_[order->order_no_] = order;
                passive_orders_queue_.push_back(order);
                auto book_cache = GetBookCacheFromPool();
                InitBookCache(book_cache);
                RefreshBookCache(book_cache, side);
                order_to_prev_book_cache_map_[order] = book_cache;
            }

            break;
        }
        default:
            break;
        }
        aggressive_order_size_pairs_.pop();
    }
}

void TaifexSimulator::ProcessPassiveOrderQueue(const Timestamp &event_loop_time)
{
    const auto &backward_delay = match_time_ + backward_delay_[Passive];
    for (auto it = passive_orders_queue_.begin(); it != passive_orders_queue_.end();)
    {
        const bool &is_bid = (*it)->side_ == TAIFEX_ORDER_SIDE::BUY;
        if ((*it)->order_received_by_exch_ts_ > event_loop_time)
        {
            return;
        }
        if ((*it)->is_need_to_be_deleted_)
        {
            if ((*it)->is_need_to_be_cancelled_)
            {
                InsertOrderReportWithTimer((*it)->price_, 0, (*it)->remaining_qty_, (*it)->side_,
                                           (*it)->timeinforce_, (*it)->position_effect_,
                                           OrderReportType::OrderCancelled, (*it)->order_no_,
                                           (*it)->order_cancel_sent_ts_ + forward_delay_[Passive] +
                                               backward_delay_[Passive]);
            }
            DestroyObjFromPool((*it), on_exch_orders_pool_);
            it = passive_orders_queue_.erase(it);
            continue;
        }
        auto bcit = order_to_prev_book_cache_map_.find(*it);
        if (bcit == order_to_prev_book_cache_map_.end())
        {
            // could not find book cache thus cannot process passive order
            SPDLOG_ERROR("[{}] Could not find book cache to process passive order!", __func__);
            ++it;
        }
        else
        {
            const auto book_cache = bcit->second;
            auto       exec_pairs = is_bid ? PassiveMatch<BidType>((*it), book_cache)
                                           : PassiveMatch<AskType>((*it), book_cache);
            RefreshBookCache(book_cache, is_bid);
            (*it)->remaining_qty_ = InsertRodReport(
                event_loop_time + backward_delay + rod_report_delay_, (*it)->order_no_,
                (*it)->price_, (*it)->remaining_qty_, (*it)->side_, (*it)->timeinforce_,
                (*it)->position_effect_, &exec_pairs);
            if ((*it)->remaining_qty_ <= 0)
            {
                CleanPassiveOrderResources<true>((*it)->order_no_, event_loop_time, is_bid);
                DestroyObjFromPool((*it), on_exch_orders_pool_);
                it = passive_orders_queue_.erase(it);
                continue;
            }
            else
            {
                ++it;
            }
        }
    }
}

void TaifexSimulator::ProcessOrderQueueCallBack(const Timestamp event_loop_time,
                                                const Timestamp call_back_time, void *structure)
{
    ProcessAggressiveOrderQueue(event_loop_time);
    ProcessPassiveOrderQueue(event_loop_time);
}

void TaifexSimulator::OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o)
{
    curr_seqno_ = o->GetSequenceNumber();
    if (o->GetSymbol() == nullptr || o->GetSymbol() != symbol_)
    {
        return;
    }

    const auto &side = o->GetMarketByOrderSide();
    const auto &price =
        static_cast<int>(o->GetMarketByOrderPrice() * pm_ + 1e-8);  // for precision issue

    // Add Order Queue Position behind
    auto it = order_queue_position_[side].find(price);
    if (it != order_queue_position_[side].end())
    {
        for (auto &[order, queue_position] : it->second)
        {
            queue_position.last_position_[OrderPosition::After] =
                queue_position.position_[OrderPosition::After];
            queue_position.position_[OrderPosition::After] +=
                std::nearbyint(o->GetMarketByOrderQty());
        }
    }
}

void TaifexSimulator::OnPostBookDelete(const Timestamp              event_loop_time,
                                       const BookDataMessageDelete *o)
{
    curr_seqno_ = o->GetSequenceNumber();
    if (o->GetSymbol() == nullptr || o->GetSymbol() != symbol_)
    {
        return;
    }
    const auto &price = static_cast<int>(std::nearbyint(o->GetMarketByOrderPrice() * pm_));
    const auto &side  = o->GetMarketByOrderSide();
    const auto &qty   = static_cast<int>(std::nearbyint(o->GetMarketByOrderQty()));

    auto it = order_queue_position_[side].find(price);
    if (it == order_queue_position_[side].end())
    {
        return;
    }

    // Deduct Order Queue Position base on trade or cancel
    if (auto tit = trade_pairs_[side].find(price); tit != trade_pairs_[side].end())
    {
        for (auto &[order, queue_position] : it->second)
        {
            queue_position.last_position_[Before] = queue_position.position_[Before];
            queue_position.position_[Before] -= qty;
        }
    }
    else
    {
        const auto &check_side = !is_optimistic_of_queue_position_;
        for (auto &[order, queue_position] : it->second)
        {
            queue_position.last_position_[check_side]  = queue_position.position_[check_side];
            queue_position.last_position_[!check_side] = queue_position.position_[!check_side];
            // magic
            if (queue_position.position_[check_side] >= qty)
            {
                queue_position.position_[check_side] -= qty;
            }
            else
            {
                queue_position.position_[!check_side] -= qty;
            }
        }
    }
}

void TaifexSimulator::OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o)
{
    curr_seqno_ = o->GetSequenceNumber();
    if (o->GetSymbol() == nullptr || o->GetSymbol() != symbol_)
        return;

    const auto &side        = o->GetTradeSide();
    const auto &trade_price = static_cast<int>(std::nearbyint(o->GetTradePrice() * pm_));
    int         trade_qty   = static_cast<int>(o->GetTradeQty() + 1e-8);
    trade_pairs_[!side].insert({trade_price, trade_qty});
    UpdateQtyTakenOnBook(!side, trade_price, -trade_qty);

    switch (mode_)
    {
    case ImmediatelyFilled:
    case FilledByFillModel:
    case FilledByBookDiff:
        return;
    case FilledByBookDiffAndTrade:
        possible_filled_trades_[!side].push_back({{trade_price, trade_qty}, event_loop_time});
        return;
    default:
        return;
    }
}

void TaifexSimulator::OnPacketEnd(const Timestamp                 event_loop_time,
                                  const BookDataMessagePacketEnd *o)
{
    if (o->GetSymbol() == nullptr || o->GetSymbol() != symbol_)
    {
        return;
    }

    if (o->last_market_data_type_ != MarketDataMessageType_Trade)
    {
        trade_pairs_[Ask].clear();
        trade_pairs_[Bid].clear();
    }

    touch_price_[Bid] =
        static_cast<int>(std::nearbyint(book_->GetTouchLevel(BidType::GetType())->price_ * pm_));
    touch_price_[Ask] =
        static_cast<int>(std::nearbyint(book_->GetTouchLevel(AskType::GetType())->price_ * pm_));

    // before process we should check whether to reset market impact

    switch (market_impact_reset_mode_)
    {
    case MarketImpactResetMode::Off:
    case MarketImpactResetMode::NoImpact:
        break;
    case MarketImpactResetMode::SingleTickReset:
        if (book_->IsFlip())
        {
            CleanQtyTakenOnBook<BidType>();
            CleanQtyTakenOnBook<AskType>();
        }
        break;
    default:
        break;
    }


    switch (timer_mode_)
    {
    case TimerMode::PacketEnd:
        ProcessAggressiveOrderQueue(event_loop_time);
        ProcessPassiveOrderQueue(event_loop_time);
        break;
    case TimerMode::AccurateTimer:
    default:
        break;
    }
}

TaifexSimulationOrderManager::TaifexSimulationOrderManager(MultiBookManager *multi_book_manager,
                                                           ObjectManager *   object_manager,
                                                           Engine *engine, size_t config_index)
    : TaifexOrderManagerBase(object_manager)
    , multi_book_manager_{multi_book_manager}
    , object_manager_{object_manager}
    , universe_{multi_book_manager_->GetUniverse()}
    , symbol_numbers_{universe_.size()}
    , order_report_listeners_{GetOrderReportListener()}
    , order_count_{0}
    , engine_{engine}
    , curr_orderno_{'0', '0', '0', '0', '0', '\0'}
{
    // dummy configs for each symbol in exchange simulators
    auto config{object_manager_->GetGlobalConfiguration(config_index)};
    for (size_t i = 0; i < symbol_numbers_; ++i)
    {
        SPDLOG_INFO("am i here?12");
        const auto &symbol = universe_[i];
        const auto &sid    = symbol->GetDataSourceID();
        // skip symbols that does not have book.
        if (sid == DataSourceID::TPRICE || sid == DataSourceID::END)
        {
            SPDLOG_INFO("[{}] is skipped from simulators.", symbol->to_string());
            continue;
        }

        symbol_to_id_maps_[symbol->GetDataSourcePid()] = i;
        taifex_simulators_.push_back(
            new TaifexSimulator(config, symbol, multi_book_manager_, engine_, this));
        order_status_pools_.push_back(new boost::pool<boost::default_user_allocator_malloc_free>(
            sizeof(TaifexOrderStatus), 256));
    }
}

TaifexSimulationOrderManager::~TaifexSimulationOrderManager()
{
    for (auto &t : taifex_simulators_)
    {
        if (t)
        {
            delete t;
            t = nullptr;
        }
    }

    for (auto &osp : order_status_pools_)
    {
        if (osp)
        {
            delete osp;
            osp = nullptr;
        }
    }
}

void TaifexSimulationOrderManager::SetUpSimulatorListeners()
{
    for (auto s : taifex_simulators_)
    {
        multi_book_manager_->AddPostBookListener(s);
    }
}

TaifexOrderStatus *TaifexSimulationOrderManager::NewFutureOrder(
    const char *pid, int price, int qty, TAIFEX_ORDER_SIDE side,
    TAIFEX_ORDER_TIMEINFORCE timeInForce, TAIFEX_ORDER_POSITIONEFFECT positionEffect,
    int sessionIndex, int subSessionIndex, unsigned int account, char accountFlag,
    const char *pUserDefine)
{
    auto sit = symbol_to_id_maps_.find(std::string(pid));
    if (BRANCH_UNLIKELY(sit == symbol_to_id_maps_.end()))
    {
        SPDLOG_ERROR("Cannot find {} in TaifexSimulator", pid);
        return nullptr;
    }
    const auto &id       = sit->second;
    const auto &order_no = ++order_count_;
    taifex_simulators_[id]->NewOrder(order_no, pid, price, qty, side, timeInForce, positionEffect,
                                     engine_->GetCurrentTime());
    auto *order_status = GetOrderStatusFromPoolWithId(id);
    strcpy(order_status->OrderNo, IntToOrderNo(order_no).c_str());

    return order_status;
}

void TaifexSimulationOrderManager::CancelFutureOrder(const char *orderno, const char *pid,
                                                     TAIFEX_ORDER_SIDE side, int sessionIndex,
                                                     int subSessionIndex, const char *pUserDefine)
{
    auto sit = symbol_to_id_maps_.find(std::string(pid));
    if (BRANCH_UNLIKELY(sit == symbol_to_id_maps_.end()))
    {
        SPDLOG_ERROR("Cannot find {} in TaifexSimulator", pid);
        return;
    }
    const auto &id       = sit->second;
    const auto &order_no = OrderNoToInt(orderno);
    taifex_simulators_[id]->CancelOrder(order_no, pid, side, true, engine_->GetCurrentTime());
}

TaifexOrderStatus *TaifexSimulationOrderManager::NewFutureDoubleOrder(
    const char *pid, int bidprice, int bidqty, int askprice, int askqty,
    TAIFEX_ORDER_TIMEINFORCE timeInForce, TAIFEX_ORDER_POSITIONEFFECT positionEffect,
    int sessionIndex, int subSessionIndex, unsigned int account, char accountFlag,
    const char *pUserDefine)
{
    auto sit = symbol_to_id_maps_.find(std::string(pid));
    if (BRANCH_UNLIKELY(sit == symbol_to_id_maps_.end()))
    {
        SPDLOG_ERROR("Cannot find {} in TaifexSimulator", pid);
        return nullptr;
    }
    const auto &id       = sit->second;
    const auto &order_no = ++order_count_;
    taifex_simulators_[id]->NewDoubleOrder(order_no, pid, bidprice, bidqty, askprice, askqty,
                                           timeInForce, positionEffect, engine_->GetCurrentTime());
    auto *order_status = GetOrderStatusFromPoolWithId(id);
    strcpy(order_status->OrderNo, IntToOrderNo(order_no).c_str());
    return order_status;
}

void TaifexSimulationOrderManager::CancelFutureDoubleOrder(const char *orderno, const char *pid,
                                                           int sessionIndex, int subSessionIndex,
                                                           const char *pUserDefine)
{
    auto sit = symbol_to_id_maps_.find(std::string(pid));
    if (BRANCH_UNLIKELY(sit == symbol_to_id_maps_.end()))
    {
        SPDLOG_ERROR("Cannot find {} in TaifexSimulator", pid);
        return;
    }
    const auto &id = sit->second;
    taifex_simulators_[id]->CancelDoubleOrder(orderno, pid, engine_->GetCurrentTime());
}


TaifexOrderStatus *TaifexSimulationOrderManager::NewOptionOrder(
    const char *pid, int price, int qty, TAIFEX_ORDER_SIDE side,
    TAIFEX_ORDER_TIMEINFORCE timeInForce, TAIFEX_ORDER_POSITIONEFFECT positionEffect,
    int sessionIndex, int subSessionIndex, unsigned int account, char accountFlag,
    const char *pUserDefine)
{
    return NewFutureOrder(pid, price, qty, side, timeInForce, positionEffect, sessionIndex,
                          subSessionIndex, account, accountFlag, pUserDefine);
}

void TaifexSimulationOrderManager::CancelOptionOrder(const char *orderno, const char *pid,
                                                     TAIFEX_ORDER_SIDE side, int sessionIndex,
                                                     int subSessionIndex, const char *pUserDefine)
{
    CancelFutureOrder(orderno, pid, side, sessionIndex, subSessionIndex, pUserDefine);
}

TaifexOrderStatus *TaifexSimulationOrderManager::NewOptionDoubleOrder(
    const char *pid, int bidprice, int bidqty, int askprice, int askqty,
    TAIFEX_ORDER_TIMEINFORCE timeInForce, TAIFEX_ORDER_POSITIONEFFECT positionEffect,
    int sessionIndex, int subSessionIndex, unsigned int account, char accountFlag,
    const char *pUserDefine)
{
    return NewFutureDoubleOrder(pid, bidprice, bidqty, askprice, askqty, timeInForce,
                                positionEffect, sessionIndex, subSessionIndex, account, accountFlag,
                                pUserDefine);
}

void TaifexSimulationOrderManager::ModifyOptionDoubleOrder(const char *orderno, const char *pid,
                                                           int bidprice, int askprice,
                                                           int sessionIndex, int subSessionIndex,
                                                           const char *pUserDefine)
{
    auto sit = symbol_to_id_maps_.find(std::string(pid));
    if (BRANCH_UNLIKELY(sit == symbol_to_id_maps_.end()))
    {
        SPDLOG_ERROR("Cannot find {} in TaifexSimulator", pid);
        return;
    }
    const auto &id = sit->second;
    taifex_simulators_[id]->ModifyOrder(orderno, pid, bidprice, askprice,
                                        engine_->GetCurrentTime());
}

void TaifexSimulationOrderManager::CancelOptionDoubleOrder(const char *orderno, const char *pid,
                                                           int sessionIndex, int subSessionIndex,
                                                           const char *pUserDefine)
{
    auto sit = symbol_to_id_maps_.find(std::string(pid));
    if (BRANCH_UNLIKELY(sit == symbol_to_id_maps_.end()))
    {
        SPDLOG_ERROR("Cannot find {} in TaifexSimulator", pid);
    }
    const auto &id = sit->second;
    taifex_simulators_[id]->CancelDoubleOrder(orderno, pid, engine_->GetCurrentTime());
}

void TaifexSimulationOrderManager::NewOrderCallBack(const Timestamp event_loop_time,
                                                    const Timestamp call_back_time, void *structure)
{
    auto order_data = reinterpret_cast<NewOrderData *>(structure);
    switch (order_data->product_type_)
    {
    case ProductType::Future:
        NewFutureOrder(order_data->pid_, order_data->price_, order_data->qty_, order_data->side_,
                       order_data->timeinforce_, order_data->position_effect_, 0, '2', 0, 0);
        break;
    case ProductType::Perp:
        NewFutureOrder(order_data->pid_, order_data->price_, order_data->qty_, order_data->side_,
                       order_data->timeinforce_, order_data->position_effect_, 0, '2', 0, 0);
        break;
    case ProductType::Option:
        NewOptionOrder(order_data->pid_, order_data->price_, order_data->qty_, order_data->side_,
                       order_data->timeinforce_, order_data->position_effect_, 0, '2', 0, 0);
        break;
    default:
        break;
    }
}

void TaifexSimulationOrderManager::CancelOrderCallBack(const Timestamp event_loop_time,
                                                       const Timestamp call_back_time,
                                                       void *          structure)
{
    auto order_data = reinterpret_cast<CancelOrderData *>(structure);
    switch (order_data->product_type_)
    {
    case ProductType::Future:
        CancelFutureOrder(IntToOrderNo(order_data->order_no_).c_str(), order_data->pid_,
                          order_data->side_, 0, 0);
        break;
    case ProductType::Perp:
        CancelFutureOrder(IntToOrderNo(order_data->order_no_).c_str(), order_data->pid_,
                          order_data->side_, 0, 0);
        break;
    case ProductType::Option:
        CancelOptionOrder(IntToOrderNo(order_data->order_no_).c_str(), order_data->pid_,
                          order_data->side_, 0, 0);
        break;
    default:
        break;
    }
}

void TaifexSimulationOrderManager::Process(const Timestamp &event_loop_time)
{
    for (const auto &s : taifex_simulators_)
    {
        for (const auto &o : *(s->GetOrderReports()))
        {
            const auto &symbol = s->GetSymbol();
            memcpy(curr_orderno_, IntToOrderNo(std::abs(o->order_no_)).c_str(), 5);
            switch (o->type_)
            {
            case OrderReportType::OrderAccepted:
                order_report_accepted_.OrderNo = curr_orderno_;
                order_report_accepted_.Price   = std::move(o->price_);
                order_report_accepted_.Qty     = std::move(o->qty_);
                order_report_accepted_.Side    = std::move(o->side_);
                for (const auto &lis : order_report_listeners_)
                {
                    lis->OnAccepted(event_loop_time, symbol, &order_report_accepted_, nullptr);
                }
                break;
            case OrderReportType::OrderCancelFailed:
                OrderReportMessageCancelFailed ocf;
                order_report_cancel_failed_.OrderNo = curr_orderno_;
                for (const auto &lis : order_report_listeners_)
                {
                    lis->OnCancelFailed(event_loop_time, &order_report_cancel_failed_, nullptr);
                }
                break;
            case OrderReportType::OrderCancelled:
                order_report_cancelled_.OrderNo = curr_orderno_;
                for (const auto &lis : order_report_listeners_)
                {
                    lis->OnCancelled(event_loop_time, symbol, &order_report_cancelled_, nullptr);
                }
                break;
            case OrderReportType::OrderRejected:
                order_report_rejected_.OrderNo = curr_orderno_;
                for (const auto &lis : order_report_listeners_)
                {
                    lis->OnRejected(event_loop_time, &order_report_rejected_, nullptr);
                }
                break;
            case OrderReportType::OrderExecuted:
                order_report_executed_.LeavesQty = std::move(o->leaves_qty_);
                order_report_executed_.OrderNo   = curr_orderno_;
                order_report_executed_.Price     = std::move(o->price_);
                order_report_executed_.Qty       = std::move(o->qty_);
                order_report_executed_.Side      = std::move(o->side_);
                for (const auto &lis : order_report_listeners_)
                {
                    lis->OnExecuted(event_loop_time, symbol, &order_report_executed_, nullptr);
                }
                break;
            case OrderReportType::OrderModified:
                order_report_modified_.OrderNo = curr_orderno_;
                order_report_modified_.Price   = std::move(o->price_);
                order_report_modified_.Qty     = std::move(o->qty_);
                order_report_modified_.Side    = std::move(o->side_);
                for (const auto &lis : order_report_listeners_)
                {
                    lis->OnModified(event_loop_time, symbol, &order_report_modified_, nullptr);
                }
                break;
            case OrderReportType::OrderModifyFailed:
                order_report_modify_failed_.OrderNo = curr_orderno_;
                for (const auto &lis : order_report_listeners_)
                {
                    lis->OnModifyFailed(event_loop_time, &order_report_modify_failed_, nullptr);
                }
                break;
            default:
                break;
            }

            DestroyObjFromPool<TaifexSimulationOrderReportMessage>(o, s->GetOrderReportsPool());
        }
        // clean the vector for each process
        s->GetOrderReports()->clear();
    }
}

void TaifexSimulationOrderManager::ProcessCallBack(const Timestamp event_loop_time,
                                                   const Timestamp call_back_time, void *structure)
{
    Process(event_loop_time);
}

}  // namespace alphaone
