#ifndef _CRYPTOMARKETDATAMESSAGEFILEREADER_H_
#define _CRYPTOMARKETDATAMESSAGEFILEREADER_H_

#include "infrastructure/common/datetime/Timestamp.h"
// #include "infrastructure/common/protobuf/MarketDataMessage.pb.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/typedef/Typedefs.h"

#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <zlib.h>

namespace alphaone
{
class CryptoMarketDataMessageFileReader
{
  public:
    CryptoMarketDataMessageFileReader(const std::string &filename);
    ~CryptoMarketDataMessageFileReader();

    bool ReadNext();  // will return true as long as there's a new item to read

    DataSourceID                       GetDataSourceID() const;
    // const protobuf::MarketDataMessage &GetMarketDataMessage() const;
    const nlohmann::json &GetMarketDataMessage() const;
    

    const std::string &GetFilename();
    Timestamp          PeekTimestamp() const;

  private:
    const std::string filename_;

    MarketDataFileStruct *      market_data_file_message_;
    // protobuf::MarketDataMessage protobuf_message_;
    nlohmann::json protobuf_message_;
    // FILE *                      file_;
    std::ifstream               file_;
    // gzFile_s *                  gzfile_;
    bool                        is_eof_;
    MarketDataFileHeader        peeked_market_data_file_header_;
    const unsigned int          bounds_[2];
    std::stringstream           stream_;
    char *                      memory_;
    off_t                       offset_;
    const off_t                 file_offset_;
    off_t                       page_offset_;
    off_t                       length_;
};
}  // namespace alphaone

#endif
