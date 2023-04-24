import glob
import os
import pickle
from sklearn import svm
import math
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder


def train(train_dir="facerec/model/train", model_save_path="facerec/model/trained_svm_model.clf", verbose=True):
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
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []
    if os.path.isfile("facerec/model/trained_svm_model.clf"):
        print("Face Recognition Model Already trained!")
    else:
        print("Training SVM classifiers...")
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
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                            face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

        # Create and train the KNN classifier
        svm_clf = svm.SVC()  # neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        svm_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(svm_clf, f)

        return svm_clf
