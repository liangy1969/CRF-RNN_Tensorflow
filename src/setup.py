from distutils.core import setup, Extension

permuto_module = Extension('_permutohedral', sources = ['permutohedral_wrap.cxx', 'permutohedral.cpp'], include_dirs=['$PYTHON_PATH/Lib/site-packages/numpy/core/include'])

setup (name = 'permutohedral', version = '0.1', author = 'YL', ext_modules = [permuto_module], py_modules = ["permutohedral"])