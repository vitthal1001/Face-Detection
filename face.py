import cv2

trainedData=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img=cv2.imread('IMG.jpg')

grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


facecoordinates=trainedData.detectMultiScale(grayimg)

x,y,w,h=facecoordinates[0]
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)


cv2.imshow('window',img)
cv2.waitKey(0)


