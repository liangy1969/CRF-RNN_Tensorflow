# CRF-RNN_Tensorflow
Tensorflow implementation of CRF-RNN

Build instructions:

1. Download and install swig

2. Enter folder ./src/ 

3. Run command 'swig -c++ -python permutohedral.i'

4. Edit setup.py, replace '$PYTHON_PATH' with the root path of your python

5. Run 'python setup.py build_ext --inplace'

6. Copy the generated 'permutohedral.py' and '*.pyd'/'*.so' into the same folder with 'CRFRNN.py'

See main_example.py for guide.
