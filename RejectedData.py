############
# Folders for the rejected images 
############

import os
import numpy as np
import cv2 as cv

train_mainPath = 'archive/images/train'

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

            if (len(face_rect) == 0):
                    newfilePath = os.path.join(destPath,folder, filename)
                    cv.imwrite(newfilePath, img)

    cv.waitKey(0)


def main():
    createFolders(train_mainPath, 'rejectedData' )
    cleanedDataset(train_mainPath, 'rejectedData')


if __name__ == "__main__":
    main()
