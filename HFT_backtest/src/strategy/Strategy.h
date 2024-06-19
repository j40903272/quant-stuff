#ifndef _STRATEGY_H_
#define _STRATEGY_H_

#include "Utility.h"
#include "counter/MultiCounterManager.h"
#include "infrastructure/base/Book.h"
#include "infrastructure/base/BookDataListener.h"
#include "infrastructure/base/CommandListener.h"
#include "infrastructure/base/PreMarket.h"
#include "infrastructure/base/PreMarketListener.h"
#include "infrastructure/base/TaifexOrderReportListener.h"
#include "infrastructure/base/TimerListener.h"
#include "infrastructure/common/position/Position.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/Sequence.h"
#include "infrastructure/common/util/String.h"
#include "infrastructure/platform/engine/Engine.h"
#include "infrastructure/platform/manager/MultiBookManager.h"
#include "infrastructure/platform/manager/TaifexOrderManager.h"
#include "infrastructure/platform/manager/ObjectManager.h"
#include "infrastructure/platform/manager/SymbolManager.h"
#include "infrastructure/platform/reader/ReferenceDataReader.h"
#include "risk/RiskStatus.h"

#include <iostream>

namespace alphaone
{
class IdiotProof;
class Ensemble;
class Strategy;

using ExecutedMsgs = std::vector<std::pair<OrderReportMessageExecuted, Timestamp>>;

// Tactic
class Tactic : public BookDataListener,
               public TimerListener,
               virtual public TaifexOrderReportListener
{
  public:
    Tactic(const std::string &name, Ensemble *ensemble);
    Tactic(const std::string &name, const GlobalConfiguration *config, const Symbol *symbol,
           const Book *book, const size_t id = 0);
    Tactic(const std::string &name, Ensemble *ensemble, const nlohmann::json &fee);

    virtual ~Tactic() = default;

    // pre-book tick events
    virtual void OnPreBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
    {
    }

    // post-book tick events
    virtual void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o)
    {
    }
    virtual void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
    {
    }
    virtual void OnPostBookModifyWithPrice(const Timestamp                       event_loop_time,
                                           const BookDataMessageModifyWithPrice *o)
    {
    }
    virtual void OnPostBookModifyWithQty(const Timestamp                     event_loop_time,
                                         const BookDataMessageModifyWithQty *o)
    {
    }
    virtual void OnPostBookSnapshot(const Timestamp                event_loop_time,
                                    const BookDataMessageSnapshot *o)
    {
    }

    // trade tick events
    virtual void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o)
    {
    }

    // tick event after all event with the same timestamp has already been updated
    virtual void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o)
    {
    }

    // for production use
    virtual void OnSparseStop(const Timestamp &event_loop_time, const BookDataMessageSparseStop *o)
    {
    }

    virtual void OnTPrice(const Timestamp &event_loop_time, const MarketDataMessage *t)
    {
    }

    virtual void OnPreOpenSnapshot(const Timestamp &event_loop_time, const MarketDataMessage *t)
    {
    }

    // dynamically register symbol to universe
    virtual void RegisterSymbol(const std::string &symbol, const DataSourceType &type);
    virtual void RegisterSymbol(const Symbol *symbol, const DataSourceType &type);

    // taifex order report listener events
    virtual void OnAccepted(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageAccepted *o, void *packet)
    {
    }

    virtual void OnRejected(const Timestamp event_loop_time, OrderReportMessageRejected *o,
                            void *packet)
    {
    }

    virtual void OnCancelled(const Timestamp event_loop_time, const Symbol *symbol,
                             OrderReportMessageCancelled *o, void *packet)
    {
    }

    virtual void OnCancelFailed(const Timestamp event_loop_time, OrderReportMessageCancelFailed *o,
                                void *packet)
    {
    }

    virtual void OnExecuted(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageExecuted *o, void *packet)
    {
    }

    virtual void OnModified(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageModified *o, void *packet)
    {
    }

    virtual void OnModifyFailed(const Timestamp event_loop_time, OrderReportMessageModifyFailed *o,
                                void *packet)
    {
    }

    // twse order report listener events

    virtual void OnDropOrder(const Timestamp event_loop_time, const Symbol *symbol,
                             OrderReportMessageDropOrder *    o,
                             void *packet)
    {
    }

    virtual void OnRejectByServer(const Timestamp event_loop_time, const Symbol *symbol,
                                  OrderReportMessageRejectByServer *    o,
                                  void *packet)
    {
    }

    virtual void OnFastReport(const Timestamp event_loop_time, const Symbol *symbol,
                              OrderReportMessageFastReport *o)
    {
    }

    // timer event
    virtual void Timer(const Timestamp event_loop_time, const Timestamp call_back_time,
                       void *structure)
    {
    }

    // custom event
    virtual void OnCustomEvent(const Timestamp &event_loop_time, void *structure)
    {
    }

    // turning on and off
    inline bool IsOn() const
    {
        return is_on_;
    }

    virtual void TurnOn(const Timestamp event_loop_time = Timestamp::invalid())
    {
        is_on_ = true;
        OnActivated(event_loop_time);
    }

    virtual void TurnOff(const Timestamp event_loop_time = Timestamp::invalid())
    {
        is_on_ = false;
        OnDeactivated(event_loop_time);
    }

    virtual void OnActivated(const Timestamp event_loop_time);
    virtual void OnDeactivated(const Timestamp event_loop_time);

    inline const std::string &GetName() const
    {
        return name_;
    }

    inline const size_t &GetID() const
    {
        return id_;
    }

    inline const GlobalConfiguration *GetConfiguration() const

    {
        return config_;
    }

    inline const Symbol *GetSymbol() const
    {
        return symbol_;
    }

    inline const Book *GetBook() const
    {
        return book_;
    }

    inline Position *GetPosition()
    {
        return &position_;
    }

    inline Ensemble *GetEnsemble() const
    {
        return ensemble_;
    }

    void UpdatePosition(const Timestamp event_loop_time, const Symbol *symbol, const BookSide side,
                        const int price, const int qty);

    bool IsSimulation() const;

    // command utility
    virtual nlohmann::json OnCommandDumpStatus()
    {
        nlohmann::json sub_status;
        sub_status["Name"]              = GetName();
        sub_status["IsOn"]              = IsOn();
        sub_status["Posiiton"]          = GetPosition()->GetPosition();
        sub_status["GrossProfitOrLoss"] = GetPosition()->GetProfitOrLossGrossValue();
        sub_status["NetProfitOrLoss"]   = GetPosition()->GetProfitOrLossNetValue();
        sub_status["Cost"]              = GetPosition()->GetCost();
        sub_status["Trunover"]          = GetPosition()->GetTurnOver();
        return sub_status;
    }

    virtual nlohmann::json OnCommandDumpOutstandingOrder()
    {
        nlohmann::json json;
        return json;
    }

    virtual void OnCommandFlat()
    {
        return;
    }

    virtual nlohmann::json OnCommandSend(const Timestamp event_loop_time, const BookSide side,
                                         const BookPrice price, const BookQty qty)
    {
        nlohmann::json json;
        json["status"] = FromRiskStatusToString(RiskStatus::Good);
        return json;
    }

  protected:
    const std::string          name_;
    const size_t               id_;
    const GlobalConfiguration *config_;
    const Symbol *             symbol_;
    const Book *               book_;

    Position position_;

    Ensemble *ensemble_;

  private:
    bool is_on_;
};

// Ensemble
class Ensemble : public BookDataListener,
                 public TimerListener,
                 virtual public TaifexOrderReportListener
{
  public:
    Ensemble(const std::string &name, const Symbol *symbol, Strategy *strategy);
    Ensemble(const std::string &name, const GlobalConfiguration *config, const Symbol *symbol,
             const Book *book);
    Ensemble(const std::string &name, const Symbol *symbol, Strategy *strategy,
             const nlohmann::json &fee);

    virtual ~Ensemble() = default;

    // pre-book tick events
    void OnPreBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnPreBookDelete(event_loop_time, o);
        }
    }

    // post-book tick events
    virtual void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnPostBookAdd(event_loop_time, o);
        }
    }
    virtual void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnPostBookDelete(event_loop_time, o);
        }
    }
    virtual void OnPostBookModifyWithPrice(const Timestamp                       event_loop_time,
                                           const BookDataMessageModifyWithPrice *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnPostBookModifyWithPrice(event_loop_time, o);
        }
    }
    virtual void OnPostBookModifyWithQty(const Timestamp                     event_loop_time,
                                         const BookDataMessageModifyWithQty *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnPostBookModifyWithQty(event_loop_time, o);
        }
    }
    virtual void OnPostBookSnapshot(const Timestamp                event_loop_time,
                                    const BookDataMessageSnapshot *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnPostBookSnapshot(event_loop_time, o);
        }
    }

    // trade tick events
    virtual void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnTrade(event_loop_time, o);
        }
    }

    // tick event after all event with the same timestamp has already been updated
    virtual void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnPacketEnd(event_loop_time, o);
        }
    }

    // for production use
    virtual void OnSparseStop(const Timestamp &event_loop_time, const BookDataMessageSparseStop *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnSparseStop(event_loop_time, o);
        }
    }

    virtual void OnTPrice(const Timestamp &event_loop_time, const MarketDataMessage *t)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnTPrice(event_loop_time, t);
        }
    }

    virtual void OnPreOpenSnapshot(const Timestamp &event_loop_time, const MarketDataMessage *t)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnPreOpenSnapshot(event_loop_time, t);
        }
    }

    // dynamically register symbol to universe
    virtual void RegisterSymbol(const std::string &symbol, const DataSourceType &type);
    virtual void RegisterSymbol(const Symbol *symbol, const DataSourceType &type);

    // taifex order report listener events
    virtual void OnAccepted(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageAccepted *o, void *packet)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnAccepted(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnRejected(const Timestamp event_loop_time, OrderReportMessageRejected *o,
                            void *packet)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnRejected(event_loop_time, o, packet);
        }
    }

    virtual void OnCancelled(const Timestamp event_loop_time, const Symbol *symbol,
                             OrderReportMessageCancelled *o, void *packet)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnCancelled(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnCancelFailed(const Timestamp event_loop_time, OrderReportMessageCancelFailed *o,
                                void *packet)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnCancelFailed(event_loop_time, o, packet);
        }
    }

    virtual void OnExecuted(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageExecuted *o, void *packet)
    {
        UpdatePosition(event_loop_time, symbol, (o->Side == OrderReportSide::Buy ? BID : ASK),
                       o->Price, o->Qty);

        for (auto &tactic : tactics_)
        {
            tactic->OnExecuted(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnModified(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageModified *o, void *packet)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnModified(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnModifyFailed(const Timestamp event_loop_time, OrderReportMessageModifyFailed *o,
                                void *packet)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnModifyFailed(event_loop_time, o, packet);
        }
    }

    // twse order report listener events

    virtual void OnDropOrder(const Timestamp event_loop_time, const Symbol *symbol,
                             OrderReportMessageDropOrder *    o,
                             void *packet)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnDropOrder(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnRejectByServer(const Timestamp event_loop_time, const Symbol *symbol,
                                  OrderReportMessageRejectByServer *    o,
                                  void *packet)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnRejectByServer(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnFastReport(const Timestamp event_loop_time, const Symbol *symbol,
                              OrderReportMessageFastReport *o)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnFastReport(event_loop_time, symbol, o);
        }
    }


    // timer event
    virtual void Timer(const Timestamp event_loop_time, const Timestamp call_back_time,
                       void *structure)
    {
    }

    // custom event
    virtual void OnCustomEvent(const Timestamp &event_loop_time, void *structure)
    {
        for (auto &tactic : tactics_)
        {
            tactic->OnCustomEvent(event_loop_time, structure);
        }
    }

    const std::string &GetName() const
    {
        return name_;
    }

    const GlobalConfiguration *GetConfiguration() const
    {
        return config_;
    }

    const Symbol *GetSymbol() const
    {
        return symbol_;
    }

    const Book *GetBook() const
    {
        return book_;
    }

    Strategy *GetStrategy() const
    {
        return strategy_;
    }

    Position *GetPosition()
    {
        return &position_;
    }

    const std::vector<Tactic *> *GetAllTactics()
    {
        return &tactics_;
    }

    size_t GenerateID(const std::string &name) const
    {
        return ++(id_.insert({name, 0}).first->second);
    }

    void UpdatePosition(const Timestamp event_loop_time, const Symbol *symbol, const BookSide side,
                        const int price, const int qty);

    bool IsSimulation() const;

  protected:
    const std::string          name_;
    const GlobalConfiguration *config_;
    const Symbol *             symbol_;
    const Book *               book_;

    Position position_;

    Strategy *            strategy_;
    std::vector<Tactic *> tactics_;

  private:
    mutable std::map<std::string, size_t> id_;
};

// Strategy
class Strategy : public BookDataListener,
                 public TimerListener,
                 public CommandListener,
                 virtual public TaifexOrderReportListener,
                 public RingBufferListener,
                 public PreMarketListener
{
  public:
    Strategy(const std::string &name, ObjectManager *object_manager,
             MultiBookManager *multi_book_manager, MultiCounterManager *multi_counter_manager,
             Engine *engine, TaifexOrderManagerBase *taifex_order_manager);
    Strategy(const Strategy &) = delete;
    Strategy &operator=(const Strategy &) = delete;

    virtual ~Strategy();

    // pre-book tick events
    void OnPreBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnPreBookDelete(event_loop_time, o);
        }
    }

    // post-book tick events
    virtual void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnPostBookAdd(event_loop_time, o);
        }
    }
    virtual void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnPostBookDelete(event_loop_time, o);
        }
    }
    virtual void OnPostBookModifyWithPrice(const Timestamp                       event_loop_time,
                                           const BookDataMessageModifyWithPrice *o)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnPostBookModifyWithPrice(event_loop_time, o);
        }
    }
    virtual void OnPostBookModifyWithQty(const Timestamp                     event_loop_time,
                                         const BookDataMessageModifyWithQty *o)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnPostBookModifyWithQty(event_loop_time, o);
        }
    }
    virtual void OnPostBookSnapshot(const Timestamp                event_loop_time,
                                    const BookDataMessageSnapshot *o)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnPostBookSnapshot(event_loop_time, o);
        }
    }

    // trade tick events
    virtual void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnTrade(event_loop_time, o);
        }
    }

    // tick event after all event with the same timestamp has already been updated
    virtual void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o)
    {
        ++packet_end_count_;
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnPacketEnd(event_loop_time, o);
        }
    }

    // for production use
    virtual void OnSparseStop(const Timestamp &event_loop_time, const BookDataMessageSparseStop *o)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnSparseStop(event_loop_time, o);
        }
    }

    // taifex order report listener events
    virtual void OnAccepted(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageAccepted *o, void *packet)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnAccepted(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnRejected(const Timestamp event_loop_time, OrderReportMessageRejected *o,
                            void *packet)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnRejected(event_loop_time, o, packet);
        }
    }

    virtual void OnCancelled(const Timestamp event_loop_time, const Symbol *symbol,
                             OrderReportMessageCancelled *o, void *packet)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnCancelled(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnCancelFailed(const Timestamp event_loop_time, OrderReportMessageCancelFailed *o,
                                void *packet)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnCancelFailed(event_loop_time, o, packet);
        }
    }

    virtual void OnExecuted(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageExecuted *o, void *packet)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnExecuted(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnModified(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageModified *o, void *packet)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnModified(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnModifyFailed(const Timestamp event_loop_time, OrderReportMessageModifyFailed *o,
                                void *packet)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnModifyFailed(event_loop_time, o, packet);
        }
    }

    // twse order report listener events

    virtual void OnDropOrder(const Timestamp event_loop_time, const Symbol *symbol,
                             OrderReportMessageDropOrder *    o,
                             void *packet)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnDropOrder(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnRejectByServer(const Timestamp event_loop_time, const Symbol *symbol,
                                  OrderReportMessageRejectByServer *    o,
                                  void *packet)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnRejectByServer(event_loop_time, symbol, o, packet);
        }
    }

    virtual void OnFastReport(const Timestamp event_loop_time, const Symbol *symbol,
                              OrderReportMessageFastReport *o)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnFastReport(event_loop_time, symbol, o);
        }
    }

    virtual void OnDisconnected(const Timestamp &event_loop_time)
    {
        SPDLOG_WARN("The client of this strategy is disconnected");
        for (auto &ensemble : ensembles_)
        {
            for (auto &tactic : *ensemble->GetAllTactics())
            {
                tactic->TurnOff();
            }
        }
    }

    // ring buffer event
    virtual void OnRingBufferNewData(const Timestamp &event_loop_time, void *ptr)
    {
    }

    // timer event
    virtual void Timer(const Timestamp event_loop_time, const Timestamp call_back_time,
                       void *structure)
    {
    }

    virtual void OnCustomEvent(const Timestamp &event_loop_time, void *structure)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnCustomEvent(event_loop_time, structure);
        }
    }

    virtual void PreMarketCallBack(const Timestamp event_loop_time, const Timestamp call_back_time,
                                   void *structure)
    {
        auto m = reinterpret_cast<PreMarketMessage *>(structure);
        mm_    = MarketDataMessage{DataSourceType::Invalid};

        TranslateMessage(symbol_manager_, m, mm_);
        OnPreOpenSnapshot(event_loop_time, &mm_);
    }

    virtual void OnPreOpenSnapshot(const Timestamp &event_loop_time, const MarketDataMessage *t)
    {
        for (auto &ensemble : ensembles_)
        {
            ensemble->OnPreOpenSnapshot(event_loop_time, t);
        }
    }

    virtual void AddIdiotProof(IdiotProof *idiotproof)
    {
        if (idiotproof == nullptr)
        {
            return;
        }

        if (std::find(idiotproofs_.begin(), idiotproofs_.end(), idiotproof) == idiotproofs_.end())
        {
            idiotproofs_.push_back(idiotproof);
        }
    }

    virtual bool CheckIdiotProof() const
    {
        return !idiotproofs_.empty();
    }

    virtual void Response(const std::string &response)
    {
        // if (command_relay_)
        // {
        //     command_relay_->Response(response);
        // }
    }

    virtual void UpdateLastHeartbeat(const Timestamp &event_loop_time)
    {
        last_heartbeat_ts_ = event_loop_time;
    }

    virtual void RegisterSymbol(const std::string &symbol, const DataSourceType &type)
    {
        RegisterSymbol(symbol_manager_->GetSymbolByString(symbol), type);
    }

    virtual void RegisterSymbol(const Symbol *symbol, const DataSourceType &type)
    {
        multi_book_manager_->AddSymbolToUniverse(symbol, type);
    }

    virtual void TurnOff(const Timestamp event_loop_time = Timestamp::invalid())
    {
        is_on_ = false;
    }

    virtual void TurnOn(const Timestamp event_loop_time = Timestamp::invalid())
    {
        is_on_ = true;
    }

    virtual bool IsOn()
    {
        return is_on_;
    }

    // command event
    // virtual void OnCommand(const Timestamp &                    event_loop_time,
    //                        const MessageFormat::GenericMessage &gm);
    virtual void OnCommandHelper(const Timestamp &event_loop_time, const std::string &word);

    // getter
    inline const std::string &GetName() const
    {
        return name_;
    }

    inline const ObjectManager *GetObjectManager() const
    {
        return object_manager_;
    }

    inline const GlobalConfiguration *GetConfiguration() const
    {
        return config_;
    }

    inline const SymbolManager *GetSymbolManager() const
    {
        return symbol_manager_;
    }

    inline MultiBookManager *GetMultiBookManager() const
    {
        return multi_book_manager_;
    }

    inline MultiCounterManager *GetMultiCounterManager() const
    {
        return multi_counter_manager_;
    }

    inline Engine *GetEngine() const
    {
        return engine_;
    }

    inline TaifexOrderManagerBase *GetTaifexOrderManager() const
    {
        return taifex_order_manager_;
    }

    inline const std::vector<Ensemble *> *GetEnsembles()
    {
        return &ensembles_;
    }

    inline bool IsSimulation() const
    {
        return GetEngine()->IsSimulation();
    }

    inline const Timestamp &GetLastHeartbeatTime() const
    {
        return last_heartbeat_ts_;
    }

    inline size_t GetPacketEndCount() const
    {
        return packet_end_count_;
    }

    inline const Counter *GetCounter(const nlohmann::json &node, const Symbol *s) const
    {
        if (!multi_counter_manager_)
            return nullptr;
        return multi_counter_manager_->GetCounter(
            s, node.value("counter_interval", "SingleTickInterval"));
    }

  protected:
    const std::string          name_;
    const ObjectManager *      object_manager_;
    const GlobalConfiguration *config_;
    const SymbolManager *      symbol_manager_;
    MultiBookManager *         multi_book_manager_;
    MultiCounterManager *      multi_counter_manager_;
    Engine *                   engine_;
    TaifexOrderManagerBase *   taifex_order_manager_;

    std::vector<Ensemble *>   ensembles_;
    std::vector<IdiotProof *> idiotproofs_;

    const std::string heartbeat_string_;
    Timestamp         last_heartbeat_ts_;
    bool              is_on_;

    size_t packet_end_count_;

    const Symbol *    registered_symbol_;
    Tactic *          switched_tactic_;
    MarketDataMessage mm_;
};
}  // namespace alphaone

#endif
