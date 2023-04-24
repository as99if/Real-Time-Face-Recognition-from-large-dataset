"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:


* in predict and show result func. 
    if model_save_path == trained_knn_model.clf then knn ---- threshold distance .6 for better classification
    if model_save_path == trained_svm_model.clf then svm  --- not suitable as there is no tolerance or threshold to set
                                                                no unknown result
    if model_save_path == trained_bayes_model.clf then GaussianNB --- missclassifies as other name, 
                                                                    no Unknown result
NOTE: This example requires some packages to be installed:
#install pip
$ pip install scikit-learn
$ pip install numpy
$ pip install scipy
# install face_recognition package


-first looks for classifier
-if classifier not found, train (from train folder)
-gets embedding or face_encodings array from camera frame
-puts to predict function
-show result and sends REST cmd to server for sending notification to client


"""

import pickle

import cv2
import face_recognition
from facerec.model import model_train
import logging

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class FaceRec:
    def __init__(self, video_capture):

        logging.basicConfig(encoding='utf-8', level=logging.INFO,
                            datefmt='%d-%b-%y %H:%M:%S',
                            format='%(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler("log.log"),
                                logging.StreamHandler()
                            ]
                            )
        # STEP 1: Train the classifier and save it to disk
        # Once the model is trained and saved, you can skip this step next time.
        svm_classifier = model_train.train()
        logging.info("Training complete!")

        # STEP 2: face detection from video, saves snap with faces to test directory
        # from real time video feed
        self.face_rec(video_capture, svm_classifier)

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def predict(self, face_encoding_array, clf=None, model_path="facerec/model/trained_svm_model.clf",
                distance_threshold=0.6):
        """
        Recognizes faces in given image using a trained KNN classifier

        :param X_img_path: path to image to be recognized
        :param clf: (optional) a classifier object. if not specified, model_save_path must be specified.
        :param model_path: (optional) path to a pickled knn or svm classifier. if not specified, model_save_path must be clf.
        :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
               of mis-classifying an unknown person as a known one.
        :return: a list of names for the recognized faces in the image: [(name, ...].
            For faces of unrecognized persons, the name 'unknown' will be returned.
        """

        if clf is None and model_path is None:
            raise Exception("Must supply classifier either through clf or model_path")

        # Load a trained model (if one was passed in)
        if clf is None:
            with open(model_path, 'rb') as f:
                clf = pickle.load(f)

        # -------------------------- svm ----------------------------#
        result = clf.predict([face_encoding_array[0]])
        return result
        # -----------------------------------------------------------------------#

    def show_prediction_labels_on_image(self, predictions, frame, face_locations):
        """
        Shows the face recognition results visually.

        :param face_locations: detected face location
        :param frame: video frame
        :param img_path: path to image to be recognized
        :param predictions: results of the predict function
        :return:
        """

        for name, (top, right, bottom, left) in zip(predictions, face_locations):
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            logging.info(str(name) + " - " + str(face_locations))
            cv2.putText(frame, name, (left + 6, bottom + 25), font, 1.0, (255, 255, 255), 1)

    def face_rec(self, video_capture, clf):
        process_this_frame = True
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)

            if len(face_locations) > 0:
                # timestamp_capture = long(time.time() * 1000.0)
                face_encoding = face_recognition.face_encodings(face_image=rgb_small_frame,
                                                                known_face_locations=face_locations, model="large")

                # STEP 3: Using the trained classifier, make predictions for unknown images
                predictions = self.predict(face_encoding, clf=clf)

                # STEP 4: Shows and saves image with recognition
                self.show_prediction_labels_on_image(predictions, small_frame, face_locations)

            # process_this_frame = not process_this_frame
            # Display the resulting image
            cv2.imshow('Video', small_frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
