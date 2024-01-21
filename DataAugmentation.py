####
#   Augment the Cleaned data for both frontal only as well as frontal & profile
#   Augmented Frontal Datasets => augmented_train_frontal
#   Augmented Profile and Frontal Datasets => augmented_train_both
####
import os
import cv2 as cv
import matplotlib.pyplot as plt

FOLDER_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

## create sub folders
def subFoldersCreation(mainPath, destPath): 
    for folder in os.listdir(mainPath):
        os.chdir(destPath)
        os.mkdir(folder)
        os.chdir("..")

## Create the folders and sub folders
def createNewFolders (trainMainPath, frontalOnly):
    ## get name of the new folders
    train_destPath = 'augmented_train_frontal' if frontalOnly else 'augmented_train_both' 

    # Create the folders
    os.mkdir(train_destPath)

    # Create the sub-folders
    subFoldersCreation(trainMainPath, train_destPath)

    # Add the images
    augmentImage(trainMainPath, train_destPath)
    # augmentImage(testMainPath, test_destPath)


## augment each image in the sub-folders
def augmentImage (mainPath, destPath):
    for folder in os.listdir(mainPath):
        newPath = os.path.join(mainPath, folder)
        
        for fileName in os.listdir(newPath):
            img = cv.imread(os.path.join(newPath,fileName))
            
            # Augment the image
            img_flip = cv.flip(img, 1)  # Horizontal flip
            img_flip_blur = cv.GaussianBlur(img_flip, (3,3), 0)   # Blurring the horizontally flip image
            # Blurring image using GaussianBlur
            img_blur = cv.GaussianBlur(img, (3,3), 0)

            # save the original image and the augmented ones
            newfilePath = os.path.join(destPath, folder, fileName)
            # get prefix for the names
            getPrefix = fileName.split('.jpg')[0]
            newfilePath_flip = os.path.join(destPath, folder, (getPrefix + '_flip.jpg'))
            newfilePath_flip_blur = os.path.join(destPath, folder, (getPrefix + '_flip_blur.jpg'))
            newfilePath_blur = os.path.join(destPath, folder, (getPrefix + '_blur.jpg'))

            cv.imwrite(newfilePath, img)
            cv.imwrite(newfilePath_flip, img_flip)
            cv.imwrite(newfilePath_flip_blur, img_flip_blur)
            cv.imwrite(newfilePath_blur, img_blur)

            cv.waitKey(0)



def main():
    # check if only frontal faces or both frontal and profile should be extracted
    face_type_clean = input("Frontal Only or Both (Frontal || profile): \n")
    getFrontalOnly = True if face_type_clean == "frontal" else False
    print("getFrontalOnly: ", getFrontalOnly)

    if (getFrontalOnly):
        createNewFolders('train_frontal', getFrontalOnly)
    else:
        createNewFolders('train_both', getFrontalOnly)


if __name__ == "__main__":
    main()
