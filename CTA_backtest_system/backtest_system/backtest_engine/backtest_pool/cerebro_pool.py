CRYPTO = "CRYPTO"

DEV = "DEV"
TEST = "TEST"
PROD = "PROD"

BINANCE = "binance"
UPERP = "UPERP"

PROD_CRYPTO_POOL = [

    ("SentimentAlgo", "STA_ETH_4h_a01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("SentimentAlgo", "STA_ETH_4h_a02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("SentimentAlgo", "STA_ETH_4h_a03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),

    ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_7802", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_7806", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("CompressExpAlgo", "CompressExpAlgo_ETH_1h_2635", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("TriangleSmoothAlgo", "TSMA_ETH_15m_c03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),

    ("SmoAlgo", "SMO_ETH_30m_b01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("SmoAlgo", "SMO_ETH_30m_b03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),

    ("EfficiencyRatioZscoreStrategy", "ERZ_BNB_2h_a01", BINANCE, UPERP, "BNBUSDT_UPERP", CRYPTO), 
    ("EfficiencyRatioZscoreStrategy", "ERZ_BNB_4h_7834", BINANCE, UPERP, "BNBUSDT_UPERP", CRYPTO), 
    ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_2018", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    ("SmoAlgo", "SMO_BTC_30m_a01", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),

]

TEST_POOL = [

    ("SentimentAlgo", "STA_ETH_4h_a02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_7806", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    ("TriangleSmoothAlgo", "TSMA_ETH_15m_c03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    ("SmoAlgo", "SMO_ETH_30m_b03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    ("EfficiencyRatioZscoreStrategy", "ERZ_BNB_2h_a01", BINANCE, UPERP, "BNBUSDT_UPERP", CRYPTO), 
    ("EfficiencyRatioZscoreStrategy", "ERZ_BNB_4h_7834", BINANCE, UPERP, "BNBUSDT_UPERP", CRYPTO), 
    ("SmoAlgo", "SMO_BTC_30m_a01", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO), 
]

CRYPTO_DEV_POOL = [

    # # SDA
    ("SmaDetrendingAlgo", "SDA_ETH_1h_a01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("SmaDetrendingAlgo", "SDA_ETH_1h_a02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("SmaDetrendingAlgo", "SDA_ETH_1h_a03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("SmaDetrendingAlgo", "SDA_ETH_1h_a04", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("SmaDetrendingAlgo", "SDA_ETH_1h_a05", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("SmaDetrendingAlgo", "SDA_ETH_1h_a06", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    ("SmaDetrendingAlgo", "SDA_ETH_1h_a07", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),



    # TDEZ ## Failed in b-round tuning for ETH-1h
    # ("TriangleDiffErZscoreStrategy", "TDEZ_ETH_1h_a01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("TriangleDiffErZscoreStrategy", "TDEZ_ETH_1h_a02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("TriangleDiffErZscoreStrategy", "TDEZ_ETH_1h_a03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),


    # # alpha pool
    # ("SentimentAlgo", "STA_ETH_4h_a01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SentimentAlgo", "STA_ETH_4h_a02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), ##
    # ("SentimentAlgo", "STA_ETH_4h_a03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),

    # ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_7802", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_7806", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), ##
    # ("CompressExpAlgo", "CompressExpAlgo_ETH_1h_2635", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("TriangleSmoothAlgo", "TSMA_ETH_15m_c03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), ##

    # ("SmoAlgo", "SMO_ETH_30m_b01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_ETH_30m_b03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), ##

    # ("EfficiencyRatioZscoreStrategy", "ERZ_BNB_2h_a01", BINANCE, UPERP, "BNBUSDT_UPERP", CRYPTO), ##
    # ("EfficiencyRatioZscoreStrategy", "ERZ_BNB_4h_7834", BINANCE, UPERP, "BNBUSDT_UPERP", CRYPTO), ##
    # ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_2018", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("SmoAlgo", "SMO_BTC_30m_a01", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO), ##

    #############################################################################################
    # current initial running
    # ("TriangleSmoothAlgo", "TSMA_ETH_15m_8934", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("CompressExpAlgo", "CompressExpAlgo_ETH_1h_2635", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_BTC_30m_a02", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("SqzCrossStrategy", "SQZCROSS_BTC_60m_c01", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),

    #############################################################################################
    
    # ("TriangleSmoothAlgo", "TSMA_ETH_15m_5831", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("TriangleSmoothAlgo", "TSMA_ETH_15m_7772", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("TriangleSmoothAlgo", "TSMA_ETH_15m_8934", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("TriangleSmoothAlgo", "TSMA_ETH_15m_a02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("TriangleSmoothAlgo", "TSMA_ETH_15m_a03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("TriangleSmoothAlgo", "TSMA_ETH_15m_a05", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("TriangleSmoothAlgo", "TSMA_ETH_15m_c03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 

    # ("SentimentAlgo", "STA_ETH_4h_a01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SentimentAlgo", "STA_ETH_4h_a02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SentimentAlgo", "STA_ETH_4h_a03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SentimentAlgo", "STA_ETH_4h_b01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SentimentAlgo", "STA_BTC_1h_a01", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("SentimentAlgo", "STA_BTC_1h_a02", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),

    # ("SmoAlgo", "SMO_ETH_30m_a01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_ETH_30m_a02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_ETH_30m_a03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_ETH_30m_a04", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),

    # ("SmoAlgo", "SMO_BTC_30m_a01", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_BTC_30m_a02", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_BTC_30m_a03", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_BTC_30m_a04", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),

    # ("SmoAlgo", "SMO_ETH_30m_b01", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_ETH_30m_b02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_ETH_30m_b03", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_ETH_30m_b04", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("SmoAlgo", "SMO_ETH_30m_b05", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),

    # ("EfficiencyRatioZscoreStrategy", "ERZ_ETH_1h_a02", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("EfficiencyRatioZscoreStrategy", "ERZ_BTC_4h_a03", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO), 
    # ("EfficiencyRatioZscoreStrategy", "ERZ_BNB_2h_a01", BINANCE, UPERP, "BNBUSDT_UPERP", CRYPTO), 
    # ("EfficiencyRatioZscoreStrategy", "ERZ_BNB_2h_b02", BINANCE, UPERP, "BNBUSDT_UPERP", CRYPTO),
    # ("EfficiencyRatioZscoreStrategy", "ERZ_BNB_4h_7834", BINANCE, UPERP, "BNBUSDT_UPERP", CRYPTO), 

    # ("CompressExpAlgo", "CompressExpAlgo_ETH_1h_0231", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("CompressExpAlgo", "CompressExpAlgo_ETH_1h_1123", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("CompressExpAlgo", "CompressExpAlgo_ETH_1h_2635", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("CompressExpAlgo", "CompressExpAlgo_ETH_1h_4490", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("CompressExpAlgo", "CompressExpAlgo_ETH_1h_9832", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 

    # ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_2018", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO), 
    # ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_7802", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),
    # ("CompressCrossAlgoEth", "CompressCrossAlgo_ETH_1h_7806", BINANCE, UPERP, "ETHUSDT_UPERP", CRYPTO),

    # ("CompressCrossAlgoBtc", "CompressCrossAlgo_BTC_1h_7803", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("CompressCrossAlgoBtc", "CompressCrossAlgo_BTC_1h_7881", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("CompressCrossAlgoBtc", "CompressCrossAlgo_BTC_1h_c01", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("CompressCrossAlgoBtc", "CompressCrossAlgo_BTC_1h_c05", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO,),
    # ("CompressCrossAlgoBtc", "CompressCrossAlgo_BTC_1h_c06", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("CompressCrossAlgoBtc", "CompressCrossAlgo_BTC_1h_c07", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("CompressCrossAlgoBtc", "CompressCrossAlgo_BTC_1h_c08", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),
    # ("CompressCrossAlgoBtc", "CompressCrossAlgo_BTC_1h_c09", BINANCE, UPERP, "BTCUSDT_UPERP", CRYPTO),

]

