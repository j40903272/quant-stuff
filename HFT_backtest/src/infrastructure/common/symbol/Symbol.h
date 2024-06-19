#ifndef _SYMBOL_H_
#define _SYMBOL_H_

// clang-format off
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/spdlog/fmt/ostr.h"
// clang-format on
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/typedef/Typedefs.h"

#include <string>
#include <vector>

#define SYMBOLID_LENGTH 20

namespace alphaone
{
DataSourceID GetDataSourceID(const std::string &source, ProductType type);
DataSourceID GetDataSourceID(const std::string &source, const std::string &type_str);
std::string  ProductTypeToString(ProductType type);
ProductType  StringToProductType(const std::string &type);
std::string  WarrantTypeToString(WarrantType type);
WarrantType  StringToWarrantType(const std::string &type);
ENUM_CPFlag  StringToCPFlag(const std::string &cpflag);

struct TickSize
{
    TickSize(double st, double e, double sz) noexcept : Start{st}, End{e}, Size{sz}
    {
    }
    double Start;
    double End;
    double Size;
};

enum UpDownDir
{
    Down   = 0,
    Up     = 1,
    UpDown = 2
};

class PriceUp
{
  public:
    inline static bool IsUpper(double price, double tick_price)
    {
        return price >= tick_price;
    }
    inline static bool IsLower(double price, double tick_price)
    {
        return price < tick_price;
    }
    inline static bool Value()
    {
        return true;
    }
};

class PriceDown
{
  public:
    inline static bool IsUpper(double price, double tick_price)
    {
        return price > tick_price;
    }
    inline static bool IsLower(double price, double tick_price)
    {
        return price <= tick_price;
    }
    inline static bool Value()
    {
        return false;
    }
};

class Symbol
{
  public:
    Symbol(std::string source, std::string group, std::string pid, std::string cpid,
           std::string underlying, ProductType product_type, WarrantType warrant_type,
           std::string maturitydate, int y, int m, int w, double cr, int v, ENUM_CPFlag cpflag,
           double strike, int multiplier, std::string currency, int dp,
           std::vector<TickSize> &ticksize_list, BookPrice ref_price,
           std::vector<BookPrice> &downlimits, std::vector<BookPrice> &uplimits, bool is_roll,
           const Symbol *c1, const Symbol *c2);
    Symbol()               = delete;
    Symbol(const Symbol &) = delete;
    Symbol(Symbol &&)      = default;
    Symbol &operator=(const Symbol &) = delete;
    Symbol &operator=(Symbol &&) = default;

    ~Symbol() = default;

    const std::string &GetSource() const;
    const std::string &GetGroup() const;
    const std::string &GetPid() const;
    const std::string &GetCPid() const;
    const std::string &GetUnderlying() const;
    const std::string &GetDataSourcePid() const;
    const std::string &GetRepresentativePid() const;
    const std::string &to_string() const;

    const std::string &GetMaturity() const;
    const Date &       GetMaturityDate() const;
    int                GetY() const;
    int                GetM() const;
    int                GetW() const;
    int                GetDecimalPrecision() const;
    double             GetDecimalConverter() const;
    double             GetConversionRate() const;
    int                GetVolume() const;
    int                GetMultiplier() const;
    ENUM_CPFlag        GetCPFlag() const;
    double             GetStrike() const;
    BookPrice          GetTickSize(BookPrice price, bool up) const;
    const std::string &GetCurrency() const;
    const Symbol *     GetC1Symbol() const;
    const Symbol *     GetC2Symbol() const;
    bool               IsRoll() const;

    template <typename Dir>
    BookPrice GetTickSize2(BookPrice price) const
    {
        auto start{0}, end{static_cast<int>(ticksize_list_.size() - 1)};
        while (start <= end)
        {
            auto mid{(start + end) / 2};
            if (Dir::IsUpper(price, ticksize_list_[mid].End))
            {
                start = mid + 1;
            }
            else if (Dir::IsLower(price, ticksize_list_[mid].Start))
            {
                end = mid - 1;
            }
            else
            {
                return ticksize_list_[mid].Size;
            }
        }
        return std::nan("NaN");
    }

    DataSourceID GetDataSourceID() const;
    ProductType  GetProductType() const;
    WarrantType  GetWarrantType() const;
    BookPrice    GetReferencePrice() const;

    template <UpDownDir dir>
    const std::vector<BookPrice> &GetLimit() const
    {
        return limits_[dir];
    }

    template <typename OStream>
    friend OStream &operator<<(OStream &os, const Symbol *s)
    {
        return os << "[" << s->to_string() << "]";
    }

    template <typename OStream>
    friend OStream &operator<<(OStream &os, const Symbol &s)
    {
        return os << "[" << s.to_string() << "]";
    }

  private:
    const std::string           source_;
    const std::string           group_;
    const std::string           pid_;
    const std::string           cpid_;
    const std::string           underlying_;
    const std::string           datasourcepid_;
    const ProductType           product_type_;
    const WarrantType           warrant_type_;
    const std::string           maturity_;
    const Date                  maturity_date_;
    const int                   y_;
    const int                   m_;
    const int                   w_;
    const double                conversion_rate_;
    const int                   volume_;
    const ENUM_CPFlag           cpflag_;
    const double                strike_;
    const int                   multiplier_;
    const int                   dp_;
    const double                dc_;
    const std::string           currency_;
    const std::vector<TickSize> ticksize_list_;
    const std::string           name_;
    const DataSourceID          data_source_id_;

    const BookPrice              ref_price_;
    const std::vector<BookPrice> limits_[UpDownDir::UpDown];

    const bool    is_roll_;
    const Symbol *c1_;  // c1 for roll symbol
    const Symbol *c2_;  // c2 for roll symbol
};
}  // namespace alphaone
#endif
