##########
#   Capture from Camera 
#   Then apply the final model for emotion detection
##################

import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import numpy as np


## live video: for frontal images only
def video_frontal (modelName):
    # load model
    loaded_model = keras.models.load_model(modelName)

    # check architecture
    loaded_model.summary()

    emotionLabels= ['angry', 'disgust', 'fear', 'happy', 'neutral','sad', 'surprise']
    print("Emotions: ", emotionLabels)

    # read in the xml file
    haar_cascade = cv.CascadeClassifier('haars_face.xml')

    cam = cv.VideoCapture(0)

    while True:
        # Capture the video frame
        ret, frame = cam.read()
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)       # OpenCV uses BGR ordering on their image channels instead of RGB
            
        # Find the face using Haar's Cascade
        face_rect = haar_cascade.detectMultiScale(grayFrame, scaleFactor = 1.3, minNeighbors = 2)

        # Draw rectangle around the face with the predicted emotion and score
        for (x,y,w,h) in face_rect:
            # cropped the face
            face = grayFrame[y: y+h, x: x+w]
            # resize, convert to array and change to grayscale
            img = cv.resize(face, dsize=(48,48))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)    # Create/ add an outer batch

            # make prediction and get score
            predictions = loaded_model.predict(img_array)

            score = tf.nn.softmax(predictions[0])
            emotion = emotionLabels[np.argmax(score)]
            confidence_level = format(100 * np.max(score), ".2f")
            text_display = '%s: %s'%(emotion, confidence_level)

            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            cv.putText(frame, text_display, org=(x,y-10), fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale= 0.6, 
                    color= (0,255,0), thickness= 2)
            

        # Display the resulting frame
        cv.imshow('Video Frame', frame)

        # the 'q' button is set as the quitting button you may use any desired button of your choice
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    cam.release()
    # Destroy all the windows
    cv.destroyAllWindows()

    cv.waitKey(0)


## open video camera but for model including both frontal and profile
def videoBoth (modelName):
    # load model
    loaded_model = keras.models.load_model(modelName)

    # check architecture
    loaded_model.summary()

    emotionLabels= ['angry', 'disgust', 'fear', 'happy', 'neutral','sad', 'surprise']
    print("Emotions: ", emotionLabels)

    # read in the xml file
    haar_cascade = cv.CascadeClassifier('haars_face.xml')
    haar_cascade_profile = cv.CascadeClassifier('haars_profileFace.xml')

    cam = cv.VideoCapture(0)

    while True:
        # Capture the video frame
        ret, frame = cam.read()
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
            
        # Find the face using Haar's Cascade
        face_rect = haar_cascade.detectMultiScale(grayFrame, scaleFactor = 1.3, minNeighbors = 3)
        face_rect_profile = haar_cascade_profile.detectMultiScale(grayFrame, scaleFactor = 1.3, minNeighbors = 3)

        # check if any frontal faces are detected
        if (len(face_rect) != 0):
            for (x,y,w,h) in face_rect:
                face = grayFrame[y: y+h, x: x+w]
                img = cv.resize(face, dsize=(48,48))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)    # Create/ add an outer batch

                predictions = loaded_model.predict(img_array)

                score = tf.nn.softmax(predictions[0])
                emotion = emotionLabels[np.argmax(score)]
                confidence_level = format(100 * np.max(score), ".2f")
                text_display = '%s: %s'%(emotion, confidence_level)

                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
                cv.putText(frame, text_display, org=(x,y-10), fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale= 0.6, 
                        color= (0,255,0), thickness= 2)

        elif ((len(face_rect) == 0) & (len(face_rect_profile) !=0)):
            for (x,y,w,h) in face_rect_profile:
                face = grayFrame[y: y+h, x: x+w]
                img = cv.resize(face, dsize=(48,48))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)   

                predictions = loaded_model.predict(img_array)

                score = tf.nn.softmax(predictions[0])
                emotion = emotionLabels[np.argmax(score)]
                confidence_level = format(100 * np.max(score), ".2f")
                text_display = '%s: %s'%(emotion, confidence_level)

                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
                cv.putText(frame, text_display, org=(x,y-10), fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale= 0.6, 
                        color= (0,255,0), thickness= 2)
        # Neither frontal or profile face detected
        else:
            grayFrame_flip = cv.flip(grayFrame,1)
            isProfile = haar_cascade_profile.detectMultiScale(grayFrame_flip, scaleFactor = 1.3, minNeighbors = 2)
            
            if (len(isProfile) != 0):
                for (x,y,w,h) in isProfile:
                    face = grayFrame_flip[y: y+h, x: x+w]
                    img = cv.resize(face, dsize=(48,48))
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)   

                    predictions = loaded_model.predict(img_array)

                    score = tf.nn.softmax(predictions[0])
                    emotion = emotionLabels[np.argmax(score)]
                    confidence_level = format(100 * np.max(score), ".2f")
                    text_display = '%s: %s'%(emotion, confidence_level)
                    print(x)
                    print(w)
                    cv.rectangle(frame, (630-x-w, y), (630-x, y+h), (0, 255, 0), thickness=2)
                    cv.putText(frame, text_display, org=(630-x-w,y-10), fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale= 0.6, 
                            color= (0,255,0), thickness= 2)

        # Display the resulting frame
        cv.imshow('Video Frame', frame)

        # the 'q' button is set as the quitting button you may use any desired button of your choice
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    cam.release()
    # Destroy all the windows
    cv.destroyAllWindows()

    cv.waitKey(0)


########### MAIN ###############
def main():
    print("Choose the type of model to apply on the live video.")
    frontalFace = int(input("\nModel with only frontal faces or both? (Enter the option number)\n 1. Frontal Only\n 2. Frontal and Profile\n"))
    augmented = int(input("Augmented or Non-augmented Dataset? (Enter the option number)\n 1. Augmented\n 2. Non-augmented\n"))

    modelName = ""
    if (frontalFace == 1):
        modelName = "Final_train_Augmented_model_100_epochs_frontal" if augmented == 1 else "Final_train_model_30_epochs_frontal"
        print("\nModel to use: ", modelName)
        # start the live video
        video_frontal(modelName)

    else:
        modelName = "Final_train_Augmented_model_90_epochs_both" if augmented == 1 else "Final_train_model_25_epochs_both"
        print("\nModel to use: ", modelName)
        # start the live video
        videoBoth(modelName)


if __name__ == "__main__":
    main()
