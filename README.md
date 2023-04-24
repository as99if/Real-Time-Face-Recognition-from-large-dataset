# Real-Time-Face-Recognition-from-large-dataset
Face Recognition from large dataset, using face_recognition, openCV (webcam, ipcam,video_file), dlib, sklearn

create three folders in the project directory named "train", with this structure to train image data 
 - train
      - face id (name)
           - pic
           - pic
           - pic
      - face id (name)
           - pic
           - pic
           - pic
       .........
       
       
tried using svm
 
requirement: (linux)
  - python 3.9

        virtualenv venv
        source venv/bin/activate

  - dlib 19.9 

        mkdir lib
        cd lib
        git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git
        cd  dlib
        python setup.py install --yes USE_AVX_INSTRUCTIONS
        
  - install dependencies
  
        pip install -r facerec/requirements.txt

Run
---
        python -m face_rec
      
      
 Thanks to
      - https://github.com/ageitgey/face_recognition
      - http://scikit-learn.org/stable/
      

