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
-gets embading or face_encodings array from camera frame
-puts to predict function
-show result and sends REST cmd to server for sending notification to client


"""
import cv2
import datetime
import time
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}



def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
     #   <train_dir>/
     #      <person1>/
     #           <somename1>.jpeg
     #           <somename2>.jpeg
     #      ...
     #      <person2>/
     #           <somename1>.jpeg
     #           <somename2>.jpeg
     #      ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(face_encoding_array, clf=None, model_path=None, distance_threshold=0.6):
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
        raise Exception("Must supply classifier either thourgh clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if clf is None:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)

    
    #if len(face_encoding_array) == 0:
    #    get_data_from_client()
    
    # Use the KNN model to find the best matches for the test face
    if model_path == "trained_knn_model.clf" :
        closest_distances = clf.kneighbors(face_encoding_array, n_neighbors=3,  return_distance=True)
        #print closest_distances
        
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_encoding_array))]
        #print are_matches
        #result = clf.predict(face_encoding_array)
        #print "knn predict result:"
        #print (result)
        # Predict classes and remove classifications that aren't within the threshold
        return [(pred) if rec else ("unknown") for pred, rec in zip(clf.predict(face_encoding_array), are_matches)]
    
    #-------------------------- svm ----------------------------#
    if model_path == "trained_svm_model.clf" :
        result = clf.predict(face_encoding_array[0])
        print "svm:"
        print (result)
        return result
    #-----------------------------------------------------------------------#
 
    #-------------------------- nb ----------------------------#
    if model_path == "trained_bayes_model.clf" :
        result = clf.predict(face_encoding_array[0])
        print "nb:"
        print (result)
        return result
    #-----------------------------------------------------------------------#



def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        pil_image.show()
        pil_image.save( "recognized/image"+ name + str(datetime.datetime.now()) + ".jpg")
    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    

def face_recog(video_capture):
    
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        img_path = ""
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if len(face_locations) > 0 :
            	timestamp_capture = long(time.time()*1000.0)
                face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)

                

            #put to predict
            	timestamp_afterCapture = long(time.time()*1000.0)
            	knn_predictions = predict(face_encoding, model_path="trained_knn_model.clf")
            	print knn_predictions
            	timestamp_result = long(time.time()*1000.0)

            	timestamp_difference = timestamp_result - timestamp_capture
            	print ("Time difference : " + str(timestamp_difference))
            	#svm_predictions = predict(img_path, model_path="trained_svm_model.clf")
            	#print svm_predictions
            	#nb_predictions = xnt "svm:"
			#show_prediction(img_path, svm_predictions)
			#print "nb:"
			#show_prediction(img_path, nb_predictions)
                  
			

                
            
        process_this_frame = not process_this_frame   
        # Display the resulting image
        #cv2.imshow('Video', small_frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
   
    
    video_capture = cv2.VideoCapture(0)
    #video_capture = cv2.VideoCapture('rtsp://admin:admin123@10.10.5.117:554/cam/realmonitor?channel=1&subtype=0')
    video_capture.set(3, 640)
    video_capture.set(4, 480)

    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    
            ##### if i send clf file from server to client, it will be edge computing
    
    if os.path.isfile("trained_knn_model.clf"):
        print("KNN Already trained!")
    else :
        print("Training KNN classifiers...")
        knn_classifier = train("train", model_save_path="trained_knn_model.clf", n_neighbors=1)
        print("Training complete!")

    if os.path.isfile("trained_svm_model.clf"):
        print("SVM Already trained!")
    else :
        print("Training SVM classifiers...")
        svm_classifier = train("train", model_save_path="trained_svm_model.clf")
        print("Training complete!")

    if os.path.isfile("trained_bayes_model.clf"):
        print("NB Already trained!")
    else :
        print("Training NB classifiers...")
        svm_classifier = train("train", model_save_path="trained_bayes_model.clf")
        print("Training complete!")
    
    # STEP 2: face detection from video, saves snap with faces to test directory
    # from real time video feed
   
    # STEP 3: Using the trained classifier, make predictions for unknown images
    # STEP 4: Shows and saves image with recognition  
    
    face_recog(video_capture)
       
    #Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()