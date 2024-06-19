class StrategyConfig:
    def __init__(self, strategy_class, config):
        self._config = config
        self.enrich_config()
        self._prepare_data_param = config["prepare_data_param"]
        self._trading_data_param = config["trading_data_param"]

    @property
    def config(self):
        return self._config

    @property
    def prepare_data_param(self):
        return self._prepare_data_param

    @property
    def trading_data_param(self):
        return self._trading_data_param

    def enrich_config(self):
        if 'weighting' in self._config:
            self._config['weighting'] = [float(i) for i in self._config['weighting']]