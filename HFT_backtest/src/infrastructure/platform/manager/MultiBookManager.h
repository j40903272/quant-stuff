#ifndef _MULTIBOOKMANAGER_H_
#define _MULTIBOOKMANAGER_H_

#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/platform/book/MarketByOrderBook.h"
#include "infrastructure/platform/book/MarketByPriceBook.h"
#include "infrastructure/platform/book/TWSEFileBook.h"
#include "infrastructure/platform/manager/SymbolManager.h"

namespace alphaone
{
class Engine;

class MultiBookManager
{
  public:
    MultiBookManager(const GlobalConfiguration *configuration, const SymbolManager *symbol_manager,
                     Engine *engine, OrderFactory *order_factory, LevelFactory *level_factory,
                     const DataSourceType &type = DataSourceType::MarketByOrder);
    MultiBookManager()                         = delete;
    MultiBookManager(const MultiBookManager &) = delete;
    MultiBookManager &operator=(const MultiBookManager &) = delete;
    MultiBookManager(MultiBookManager &&)                 = delete;
    MultiBookManager &operator=(MultiBookManager &&) = delete;

    ~MultiBookManager();

    void AddSymbolToUniverse(const Symbol *symbol, const DataSourceType &type);
    void AddPreBookListener(const Book *book, BookDataListener *listener);
    void AddPostBookListener(const Book *book, BookDataListener *listener);
    void AddPreBookListener(const Symbol *symbol, BookDataListener *listener);
    void AddPostBookListener(const Symbol *symbol, BookDataListener *listener);
    void AddPreBookListener(BookDataListener *listener);
    void AddPostBookListener(BookDataListener *listener);

    void OnMarketDataMessage(const MarketDataMessage *mm, void *raw_packet);

    void Lock();

    bool  IsInstantiated() const;
    bool  IsLocked() const;
    bool  IsInUniverse(const Symbol *symbol) const;
    Book *GetBook(const std::string &  symbol,
                  const DataSourceType type = DataSourceType::Invalid) const;
    Book *GetBook(const Symbol *symbol, const DataSourceType type = DataSourceType::Invalid) const;
    const std::vector<const Symbol *> &GetUniverse() const;

  private:
    void Init(const DataSourceType &type);

    void AddSymbol(const Symbol *symbol);
    void AddBook(const Symbol *symbol, const DataSourceType type, void *structure = nullptr);

    const GlobalConfiguration *configuration_;
    const SymbolManager *      symbol_manager_;
    Engine *                   engine_;
    OrderFactory *             order_factory_;
    LevelFactory *             level_factory_;

    bool is_instantiated_;
    bool is_locked_;

    std::vector<const Symbol *>        universe_;
    std::vector<std::shared_ptr<Book>> handled_books_;
    std::vector<std::unordered_map<const Symbol *, std::shared_ptr<Book>>>
        source_type_unordered_map_from_symbol_to_book_;

    const size_t default_type_;
};

}  // namespace alphaone
#endif
