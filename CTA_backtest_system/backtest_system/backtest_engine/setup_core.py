import numpy as np
from setuptools import find_packages, setup
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(np.get_include())

    def build_extensions(self):
        self.build_lib = '../' # modify this to change output directory
        _build_ext.build_extensions(self)
        
setup(
    ext_modules = cythonize(
        [
            "./backtest_engine/backtest_core/backtester.pyx",
            "./backtest_engine/backtest_core/simulate_position.pyx",
            "./backtest_engine/backtest_core/sl_param.pyx",
            "./backtest_engine/backtest_core/tp_param.pyx",
            # add strategy logic
        ],
        compiler_directives={'language_level' : "3"}
    ),
    cmdclass={'build_ext': build_ext},
)
