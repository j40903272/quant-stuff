#include "TestMakerStrategy.h"


namespace alphaone
{
TestMaker::TestMaker(Ensemble *ensemble, const nlohmann::json node)
    : Tactic{"TestMaker", ensemble}
    , mobook_{dynamic_cast<const MarketByOrderBook *>(book_)}
    , fee_rate_{GetConfiguration()->GetJson().at("Position").at("fee_rate")}
    , fee_cost_{GetConfiguration()->GetJson().at("Position").at("fee_cost")}
    , max_position_{node.at("max_position").get<double>()}
    , min_order_qty_{node.at("min_order_qty").get<double>()}
    , max_order_qty_{node.at("max_order_qty").get<double>()}
    , alpha_threshold_{-1 * node.at("alpha_threshold").get<double>(),
                       node.at("alpha_threshold").get<double>()}
    , expired_duration_{node.contains("expired_duration")
                            ? Duration::from_sec(node["expired_duration"].get<double>())
                            : Duration::from_sec(5.)}
    , insert_cancel_duration_{node.contains("insert_cancel_duration")
                                  ? Duration::from_sec(node["insert_cancel_duration"].get<double>())
                                  : Duration::from_sec(0.1)}
    , cancel_wait_duration_{node.contains("cancel_wait_duration")
                                ? Duration::from_sec(node["cancel_wait_duration"].get<double>())
                                : Duration::from_sec(0.1)}
    , metric_reporter_{GetConfiguration()->GetJson().at("MetricReporter"),
                       GetSymbol(),
                       GetBook(),
                       GetEnsemble()->GetStrategy()->GetMultiCounterManager(),
                       GetEnsemble()->GetStrategy()->GetEngine(),
                       GetPosition()}
    , tx_account_{GetConfiguration()->GetJson().contains("System")
                      ? (GetConfiguration()->GetJson()["System"].contains("account_name")
                             ? GetConfiguration()
                                   ->GetJson()["System"]["account_name"]
                                   .get<unsigned int>()
                             : 0)
                      : 0}
    , tx_account_flag_{GetConfiguration()->GetJson().contains("System")
                           ? (GetConfiguration()->GetJson()["System"].contains("account_flag")
                                  ? GetConfiguration()
                                        ->GetJson()["System"]["account_flag"]
                                        .get<std::string>()[0]
                                  : '2')
                           : '2'}
    , side_string_{"[Ask]", "[Bid]"}
    , last_prediction_{0.}
    , last_mid_price_{0.}
    , touch_price_{0., 0.}

{
    // Please Notice that reference price and UP/Down Limit are recorded started from 2020/10/29
    SPDLOG_INFO("Reference price = {}", symbol_->GetReferencePrice());
    SPDLOG_INFO("Up Limits = {}", fmt::join(symbol_->GetLimit<Up>(), ", "));
    SPDLOG_INFO("Down Limits = {}", fmt::join(symbol_->GetLimit<Down>(), ", "));
}

TestMaker::~TestMaker()
{
}

void TestMaker::OnPreBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
{
}

void TestMaker::OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o)
{
}

void TestMaker::OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
{
}

void TestMaker::OnPostBookModifyWithPrice(const Timestamp                       event_loop_time,
                                          const BookDataMessageModifyWithPrice *o)
{
}

void TestMaker::OnPostBookModifyWithQty(const Timestamp                     event_loop_time,
                                        const BookDataMessageModifyWithQty *o)
{
}

void TestMaker::OnPostBookSnapshot(const Timestamp                event_loop_time,
                                   const BookDataMessageSnapshot *o)
{
}

void TestMaker::OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o)
{
}

void TestMaker::OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(!IsOn()))
    {
        return;
    }

    if (BRANCH_UNLIKELY(!mobook_->IsValid()))
    {
        return;
    }

    if (BRANCH_UNLIKELY(mobook_->GetLastTradePrice() == 0 or mobook_->GetLastTradeQty() == 0))
    {
        return;
    }

    Evaluate(event_loop_time);
    metric_reporter_.UpdateOrderExecuted(event_loop_time, executeds_);
    metric_reporter_.UpdateCounterMetrics();
    executeds_.clear();
}


void TestMaker::Evaluate(const Timestamp event_loop_time)
{

    touch_price_[Ask] = mobook_->GetPrice(ASK);
    touch_price_[Bid] = mobook_->GetPrice(BID);

    last_mid_price_ = (touch_price_[Ask] + touch_price_[Bid]) * 0.5;
    const double predicted_price{last_mid_price_ * y_exp(last_prediction_)};
    SPDLOG_DEBUG("mid = {:.2f}, predict = {:.2f}, alpha = {:.8f}", last_mid_price_, predicted_price,
                 last_prediction_);
    CancelThenSend<BidType>(event_loop_time);
    CancelThenSend<AskType>(event_loop_time);
}

void TestMaker::Send(const Timestamp event_loop_time, const BookSide side, const BookPrice price,
                     const BookQty qty)
{
    const int send_price{static_cast<int>(price * symbol_->GetDecimalConverter() + 1e-8)};

    const TAIFEX_ORDER_SIDE  send_side{side == BID ? TAIFEX_ORDER_SIDE::BUY
                                                   : TAIFEX_ORDER_SIDE::SELL};
    const TaifexOrderStatus *report{
        GetEnsemble()->GetStrategy()->GetTaifexOrderManager()->NewFutureOrder(
            GetSymbol()->GetDataSourcePid().c_str(), send_price, qty, send_side,
            TAIFEX_ORDER_TIMEINFORCE::ROD, TAIFEX_ORDER_POSITIONEFFECT::OPEN, 0, 0, tx_account_,
            tx_account_flag_, "6666TEST")};
    outstanding_order_manager_.InsertOrder(event_loop_time, side, OrderNoToInt(report->OrderNo),
                                           price, qty, event_loop_time + expired_duration_);
}

void TestMaker::Cancel(const Timestamp &event_loop_time, const BookSide side, const char *orderno)
{
    const TAIFEX_ORDER_SIDE c_side{side == BID ? TAIFEX_ORDER_SIDE::BUY : TAIFEX_ORDER_SIDE::SELL};
    GetEnsemble()->GetStrategy()->GetTaifexOrderManager()->CancelFutureOrder(
        orderno, GetSymbol()->GetDataSourcePid().c_str(), c_side, 0, 0, "6666TEST");
}

void TestMaker::OnAccepted(const Timestamp event_loop_time, const Symbol *symbol,
                           OrderReportMessageAccepted *o, void *packet)
{
    SPDLOG_INFO("{} [{}] [{}] order={} "
                "position={:.0f} cost={:.0f} gross_pl={:.0f} net_pl={:.0f}",
                event_loop_time, name_, __func__, OrderNoToInt(o->OrderNo),
                GetPosition()->GetPosition(), GetPosition()->GetCost(),
                GetPosition()->GetProfitOrLossGrossValue(),
                GetPosition()->GetProfitOrLossNetValue());
}

void TestMaker::OnRejected(const Timestamp event_loop_time, OrderReportMessageRejected *o,
                           void *packet)
{
    outstanding_order_manager_.RemoveOrder(OrderNoToInt(o->OrderNo));
}

void TestMaker::OnCancelled(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageCancelled *o, void *packet)
{
    auto orderno = OrderNoToInt(o->OrderNo);
    outstanding_order_manager_.RemoveCancelOrder(orderno);
    SPDLOG_INFO("{} [{}] [{}] order={} "
                "position={:.0f} cost={:.0f} gross_pl={:.0f} net_pl={:.0f}",
                event_loop_time, name_, __func__, orderno, GetPosition()->GetPosition(),
                GetPosition()->GetCost(), GetPosition()->GetProfitOrLossGrossValue(),
                GetPosition()->GetProfitOrLossNetValue());
}

void TestMaker::OnCancelFailed(const Timestamp event_loop_time, OrderReportMessageCancelFailed *o,
                               void *packet)
{
    // may need to retry again
    SPDLOG_INFO("{} [{}] [{}] order={} "
                "position={:.0f} cost={:.0f} gross_pl={:.0f} net_pl={:.0f}",
                event_loop_time, name_, __func__, OrderNoToInt(o->OrderNo),
                GetPosition()->GetPosition(), GetPosition()->GetCost(),
                GetPosition()->GetProfitOrLossGrossValue(),
                GetPosition()->GetProfitOrLossNetValue());
}

void TestMaker::OnExecuted(const Timestamp event_loop_time, const Symbol *symbol,
                           OrderReportMessageExecuted *o, void *packet)
{
    auto orderno = OrderNoToInt(o->OrderNo);
    auto side    = o->Side == OrderReportSide::Buy ? BID : ASK;
    SPDLOG_INFO(
        "{} [{}] [{}] order={} side={} price={} qty={} remain_qty={} "
        "position={:.0f} cost={:.0f} gross_pl={:.0f} net_pl={:.0f}",
        event_loop_time, name_, __func__, orderno, o->Side == OrderReportSide::Buy ? "BID" : "ASK",
        o->Price / symbol->GetDecimalConverter(), o->Qty, o->LeavesQty,
        GetPosition()->GetPosition(), GetPosition()->GetCost(),
        GetPosition()->GetProfitOrLossGrossValue(), GetPosition()->GetProfitOrLossNetValue());
    outstanding_order_manager_.UpdateOrder(orderno, -o->Qty);
}

TestMakerEnsemble::TestMakerEnsemble(const Symbol *symbol, Strategy *strategy,
                                     const nlohmann::json node)
    : Ensemble("TestMakerEnsemble", symbol, strategy)
{
    tactics_.push_back(new TestMaker{this, node["TestMaker"]});
}

TestMakerEnsemble::~TestMakerEnsemble()
{
    for (auto &tactic : tactics_)
    {
        delete tactic;
    }
    tactics_.clear();
}

TestMakerStrategy::TestMakerStrategy(ObjectManager *      object_manager,
                                     MultiBookManager *   multi_book_manager,
                                     MultiCounterManager *multi_counter_manager, Engine *engine,
                                     TaifexOrderManagerBase *order_manager,
                                     size_t config_index)
    : Strategy{"TestMakerStrategy",
               object_manager,
               multi_book_manager,
               multi_counter_manager,
               engine,
               order_manager}
{
    std::cout<< "fk" << std::endl;
    for (const auto &node : object_manager->GetGlobalConfiguration(config_index)
                                ->GetJson()
                                .at("Strategy")
                                .at("TestMakerStrategy")
                                .at("Ensembles"))
    {
        std::cout<< "abc" << std::endl;
        const Symbol *symbol{
            object_manager->GetSymbolManager()->GetSymbolByString(node.at("symbol"))};
        std::cout<< "abc123" << std::endl;
        ensembles_.push_back(new TestMakerEnsemble{symbol, this, node});
        std::cout<< "abc123233123" << std::endl;

    }
}

TestMakerStrategy::~TestMakerStrategy()
{
    for (auto &ensemble : ensembles_)
    {
        delete ensemble;
    }
    ensembles_.clear();
}
}  // namespace alphaone
