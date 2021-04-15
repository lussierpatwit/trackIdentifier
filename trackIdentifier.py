import cv2
import numpy as np
import os

################################
#         Variables
directory = "Resources/F1-GPs/"
trackNames = [None]*28
compareArray = [None]*24
Tracks = np.zeros((28,250,250),np.uint8)
lenTracks = 0
counter = 0
################################


# Takes parameter img and converts to grayscale before returning a uniform 250x250 sized image
def preprocessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgSquare = cv2.resize(imgGray,(250,250))

    return imgSquare

# Takes in an image and returns the black and white outline of the track as a new image
def getOutline(img):
    outlineColor = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    isolatedOutline = np.zeros_like(outlineColor)
    imgCanny = cv2.Canny(img,50,50)
    # cv2.imshow("canny", imgCanny)
    contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # cv2.imshow("in Loop", imgCanny)
        if area > 100:
            cv2.drawContours(isolatedOutline,cnt,-1,(255,255,255),2)
            perim = cv2.arcLength(cnt,True)
    return cv2.cvtColor(isolatedOutline,cv2.COLOR_BGR2GRAY)


# Stratifies array into rows of 7 before recombining into 2d matrix of images
def toImgMatrix(array):
    final_row1 = np.concatenate(array[0:7],axis=1)
    final_row2 = np.concatenate(array[7:14],axis=1)
    final_row3 = np.concatenate(array[14:21],axis=1)
    final_row4 = np.concatenate(array[21:28],axis=1)
    finalMatrix = np.concatenate((final_row1,final_row2,final_row3,final_row4),axis=0)
    return finalMatrix

# Method to get the index of the smallest number in a list
def getMinIndex(array):
    minIndex = 0
    minDiff = 1
    for i in range(len(array)):
        if array[i] < minDiff:
            minDiff = array[i]
            minIndex = i
    return minIndex


# loops through all files in directory folder and adds all pngs to array Tracks
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        tempPath = directory + filename
        # print(tempPath)
        Tracks[counter] = preprocessing(cv2.imread(tempPath))
        trackNames[counter] = filename[0:len(filename)-4]
        # getCorners(Tracks[counter])
        counter = counter + 1

lenTracks = counter
counter = 0
userTrack = preprocessing(cv2.imread("Resources/userTrack/userTrack.jpg"))

cv2.imshow("userTrack", userTrack)
# print("len tracks: ",lenTracks)

for counter in range(lenTracks):
    compareArray[counter] = cv2.matchShapes(getOutline(Tracks[counter]),getOutline(userTrack),1,0.0)
predictionIndex = getMinIndex(compareArray)
print("Prediction: ",trackNames[predictionIndex])
cv2.imshow("Prediction Img",Tracks[predictionIndex])
cv2.imshow("user outline", getOutline(userTrack))

# print(compareArray)
# print(trackNames)

################################
#       testing methods
# cv2.imshow("test toImgMatrix", getOutline(toImgMatrix(Tracks)))
# cv2.imshow("tracks[0] Raw", Tracks[14])
# cv2.imshow("test getOutline", getOutline(userTrack))
# cv2.imshow("outline",getOutline(Tracks[10]))
################################

cv2.waitKey(0)
