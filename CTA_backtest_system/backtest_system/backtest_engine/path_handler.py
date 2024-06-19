
import os
import sys
from pathlib import Path

sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())

data_center_path = f'{user_home_path}/backtest_system/data_center'