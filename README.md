# Gaze Tracking using SVR

Several library need to be installed in order to run the scripts.

## Instalation

### UVC library
The UVC library needed is the following:

https://github.com/pupil-labs/pyuvc

install libuvc

install libjpeg-turbo -> if error: no nasm (Netwide Assembler) found -> install nasm: sudo apt-get install nasm

install cython

git clone https://github.com/pupil-labs/pyuvc

modify setup.py:

    elif platform.system() == 'Linux':\
        libs = ['rt', 'uvc','turbojpeg']\
        library_dirs += ['/usr/local/lib/']
        
Run the following script
    
    python setup.py install

### scikit-learn

scikit-learn version needed: 0.20

https://scikit-learn.org/stable/install.html\

    pip install -U scikit-learn

### pandas

Pandas version needed: 0.17.1

https://pandas.pydata.org/pandas-docs/stable/install.html\

    pip install pandas

## RUN

1) record data to train SVR:

    python pupil_tracking.py

800 data gives good results
data are saved in: pupil_coord.csv
You can change the value of the threshold if the pupil tracking is not good enough.
    
2) train the SVR:

    python my_svr.py
