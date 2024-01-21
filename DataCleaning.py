##############
#   Clean the data 
#   Only store the valid files in specific folders 
#   Frontal Datasets => train_frontal; test_frontal
#   Profile and Frontal Datasets => train_both; test_both
################
import os
import numpy as np
import cv2 as cv

train_mainPath = 'archive/images/train'
test_mainPath = 'archive/images/validation'

## create folders
def createFolders(mainPath, destPath): 
    for folder in os.listdir(mainPath):
        os.chdir(destPath)
        os.mkdir(folder)
        os.chdir("..")

### Just for Frontal faces
def cleanedDataset(mainPath, destPath):
    # read in the xml file
    haar_cascade = cv.CascadeClassifier('haars_face.xml')

    for folder in os.listdir(mainPath):
        newPath = os.path.join(mainPath,folder)

        for filename in os.listdir(newPath):
            img = cv.imread(os.path.join(newPath,filename))

            # Find the face using Haar's Cascade
            face_rect = haar_cascade.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 2)

            if (len(face_rect) != 0):
                for (x,y,w,h) in face_rect:
                    # cropped the face
                    face = img[y: y+h, x: x+w]
                    newfilePath = os.path.join(destPath,folder, filename)
                    cv.imwrite(newfilePath, face)
            else:
                print("Not Saving", filename, folder)

    cv.waitKey(0)


### For both Frontal and Profile faces
def cleanedDatasetBoth(mainPath, destPath):
    # read in the xml file
    haar_cascade = cv.CascadeClassifier('haars_face.xml')
    haar_cascade_profile = cv.CascadeClassifier('haars_profileFace.xml')

    for folder in os.listdir(mainPath):
        newPath = os.path.join(mainPath,folder)

        for filename in os.listdir(newPath):
            img = cv.imread(os.path.join(newPath,filename))

            # Find the face using Haar's Cascade
            face_rect = haar_cascade.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 2)
            face_rect_profile = haar_cascade_profile.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 2)

            if (len(face_rect) != 0):
                for (x,y,w,h) in face_rect:
                    # cropped the face
                    face = img[y: y+h, x: x+w]
                    newfilePath = os.path.join(destPath,folder, filename)
                    cv.imwrite(newfilePath, face)

            elif ((len(face_rect) == 0) & (len(face_rect_profile) !=0)):
                newfilePath = os.path.join(destPath,folder, filename)
                cv.imwrite(newfilePath, img)
            
            elif (len(face_rect_profile) == 0):
                img_flip = cv.flip(img,1)
                isProfile = haar_cascade_profile.detectMultiScale(img_flip, scaleFactor = 1.3, minNeighbors = 2)
                
                if (len(isProfile) != 0):
                    newfilePath = os.path.join(destPath,folder, filename)
                    cv.imwrite(newfilePath, img)
                else:
                    print("Not Saving", filename, folder)

    cv.waitKey(0)


## Get destination filename
def destFileName(frontalOnly):
    train_destPath = 'train_frontal' if frontalOnly else 'train_both' 
    test_destPath = 'test_frontal' if frontalOnly else 'test_both' 
    return (train_destPath, test_destPath)


def main():
    # check if only frontal faces or both frontal and profile should be extracted
    face_type_clean = input("Frontal Only or Both (Frontal || profile): \n")
    getFrontalOnly = True if face_type_clean == "frontal" else False
    print("getFrontalOnly: ", getFrontalOnly)

    ## get file name
    fileNames = destFileName(getFrontalOnly)
    train_destPath = fileNames[0]
    test_destPath = fileNames[1]

    if (getFrontalOnly):
        # Train Set
        createFolders(train_mainPath, train_destPath)
        cleanedDataset(train_mainPath, train_destPath)
        # Test Set
        createFolders(test_mainPath, test_destPath)
        cleanedDataset(test_mainPath, test_destPath)

    else:
        #### For Both ########
        # Train Set
        createFolders(train_mainPath, train_destPath)
        cleanedDatasetBoth(train_mainPath, train_destPath)

        # Test Set
        createFolders(test_mainPath, test_destPath)
        cleanedDatasetBoth(test_mainPath, test_destPath)


if __name__ == "__main__":
    main()
