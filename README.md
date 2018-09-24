# Real-Time-Face-Recognition-from-large-dataset
Face Recognition from large dataset, using face_recognition, openCV (webcam, ipcam,video_file), dlib, sklearn
requirement:
  - ubuntu
  - python
  - dlib (ubuntu 16.04)
  RUN in terminal
      --------------------------------------------------------
       apt-get install python-setuptools 
       git clone https://github.com/davisking/dlib.git
       cd dlib
       mkdir build; cd build; cmake ..; cmake --build .
       cd ..
       python3 setup.py install
      ------------------------------------------------------
    if error (ubuntu 14.04)
  RUN in terminal
        ---------------------------------------
        apt-get install -y --fix-missing \
        build-essential \
        cmake \
        gfortran \
        git \
        wget \
        curl \
        graphicsmagick \
        libgraphicsmagick1-dev \
        libatlas-dev \
        libavcodec-dev \
        libavformat-dev \
        libgtk2.0-dev \
        libjpeg-dev \
        liblapack-dev \
        libswscale-dev \
        pkg-config \
        python3-dev \
        python3-numpy \
        software-properties-common \
        zip \
        && apt-get clean && rm -rf /tmp/* /var/tmp/*
        --------------------------------------------
    
   RUN in terminal
        ---------------------------------------------------------------------------------------
      $ cd ~ && \
        mkdir -p dlib && \
        git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
        cd  dlib/ && \
        python3 setup.py install --yes USE_AVX_INSTRUCTIONS
        ----------------------------------------------------------------------------------------
  - face_recognition 
   RUN in terminal
      $ pip3 install face_recognition  or  pip install face_recognition
   
      $ pip install scikit-learn
      $ pip install numpy
      $ pip install scipy
      
      
      
 Thanks to - 
  https://github.com/ageitgey/face_recognition
  http://scikit-learn.org/stable/
