./protoc -I./ -I/usr/local/include --cpp_out=./ ./MarketDataMessage.proto
./protoc -I./ -I/usr/local/include --cpp_out=./ ./MessageFormat.proto
mv MarketDataMessage.pb.cc MarketDataMessage.pb.cpp
mv MessageFormat.pb.cc MessageFormat.pb.cpp