########
#   Get a visualisation for post and pre-cleaned up data
#######

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# Get the count of files in each datasets per subfolders
def cleanDatasetsCount(folderPaths):
    folderNaming = ['Original Dataset', 'Frontal Dataset', 'Profile and Frontal Dataset']
    datasetsCounts = {}

    for i,path in enumerate(folderPaths):
        # Count files in each of the subfolders
        path_counts = []
        for subfolder in os.listdir(path):
            subfolder_path = os.path.join(path, subfolder)
            # start the count
            count = 0
            for file in os.listdir(subfolder_path):
                count += 1

            path_counts.append(count)

        datasetsCounts[folderNaming[i]] = path_counts

    return(datasetsCounts)


# Function to visualise the proportion of files within each datasets per emotions
def datasetsPlot():
    folderPaths_train = ["archive/images/train", "train_frontal", 'train_both']
    folderPaths_test = ["archive/images/validation", "test_frontal", 'test_both']

    fileName = input("Which dataset should be plotted? (Train or Test)\n")

    path = folderPaths_train if fileName.lower() == "train" else folderPaths_test

    counts = cleanDatasetsCount(path)
    print(counts)
    
    count_df = pd.DataFrame(counts,index=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
    std_count_df = count_df/np.sum(count_df,axis=0)
    print(std_count_df)

    # get the proportion of labels within the dataset; Proportion of the dataset is still relatively the same
    std_count_df.plot(kind="bar", figsize=(8, 8))
    datasetType = "training" if fileName.lower() == "train" else "testing"
    plt.title(f"Emotions proportions per {datasetType} datasets")
    plt.xlabel("Emotions")
    plt.xticks(rotation = 360) 
    plt.savefig(fileName)
    plt.show()
    
    return


# Get a table with the images 
def rejectedImgTable():
    titles = ['Occlusions', 'Non Facial Images', 'Cropped Out Images', 'Facial Poses']
    imageNames = ['occlusion', 'nonFace', 'CroppedOutFace', 'pose']
    imageList= []
    
    for j in imageNames:
        for i in range(4):
            imageList.append("rejectedData/toDisplay/%s%s.jpg"%(j, i+1))

    print(imageList)

    mainFigure = plt.figure()
    mainFigure.subplots_adjust(top= 0.8, bottom= 0.2)
    subfigures = mainFigure.subfigures(nrows= 4 , ncols= 1)

    for row, fig in enumerate(subfigures):
        fig.suptitle(titles[row], fontsize= 14, y= 1)
        # create the plots
        img = fig.subplots(nrows= 1, ncols= 4)
        for ind, image in enumerate(img):
            res = imageList[4*row+ind]
            # display the image
            res= plt.imread(res)
            image.imshow(res)
            image.axis('off')

    plt.show()
    return


# Get a table for the cleaned images per emotions
def cleanedImgTable():
    path = "Results/SampleCleanedImg/"
    emotions = []
    imageList_frontal= []
    imageList_profile= []

    for folder in os.listdir(path):
        emotions.append(folder)
        newPath = os.path.join(path, folder)
        
        for file in os.listdir(newPath):
            if (file.startswith("f")):
                imageList_frontal.append(path+folder+"/"+file)
            else:
                imageList_profile.append(path+folder+"/"+file)

    print(emotions)
    print(imageList_frontal)
    print(imageList_profile)

    # Make the figure plot
    mainFigure = plt.figure()
    # mainFigure.subplots_adjust(top= 5, bottom= 1.2)
    subfigures = mainFigure.subfigures(nrows= 8 , ncols= 3)

    subfigures[0,0].suptitle("Emotions", fontsize=14)
    subfigures[0,1].suptitle("Frontal Faces", fontsize=14)
    subfigures[0,2].suptitle("Profile Faces", fontsize=14)

    ## start the table
    for i in range(7):
        for j in range(3):
            ind = i+1
            # if in column 1, emotion type
            if (j == 0):
                subfigures[ind,0].suptitle(emotions[i], fontsize = 12)
            
            # split in 2 column
            elif (j > 0):
                imageSubplots = subfigures[ind,j].subplots(nrows = 1, ncols = 2)
                for col, img in enumerate(imageSubplots):
                    image = ""
                    # frontal Images
                    if (j == 1):
                        image = imageList_frontal[i*2 + col]
                    else:
                        image = imageList_profile[i*2 + col]

                    # display the image
                    res= plt.imread(image)
                    img.imshow(res)
                    img.axis('off')

    plt.show()
    return


# Sample plot for augmentations
def augmentedImagePlot():
    img = cv.imread('train_frontal/neutral/5927.jpg')
    plt.subplot(141)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original")

    ## Blur
    img_blur = cv.GaussianBlur(img, (3,3), 0)
    plt.subplot(142)
    plt.imshow(img_blur)
    plt.axis('off')
    plt.title("Blur")

    flip_horizontal=cv.flip(img, 1) # Horizontal flip
    plt.subplot(143)
    plt.imshow(flip_horizontal)
    plt.axis('off')
    plt.title("Flip")

    ## Blur
    img_flipBlur = cv.GaussianBlur(flip_horizontal, (3,3), 0)
    plt.subplot(144)
    plt.imshow(img_blur)
    plt.axis('off')
    plt.title("Flip and Blur")

    plt.show()
    cv.waitKey(0)


def main():

    plotToGet = int(input("Select the plot to get by entering the option number. \n\n1. Emotions proportions per datasets \n2. A table of rejected images \n 3. A table with the cleaned data \n4. Sample augmented images plot\n\n"))

    if plotToGet == 1:
        datasetsPlot() 
    elif plotToGet == 2:
        rejectedImgTable() 
    elif plotToGet == 3:
        cleanedImgTable()
    elif plotToGet == 4:
        augmentedImagePlot()
    return


if __name__=="__main__":
    main()
