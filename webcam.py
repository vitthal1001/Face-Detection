import cv2
trainedData=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam=cv2.VideoCapture(0)

while True:
    success,img=webcam.read()

    #cv2.imshow('pic2',img)
    #cv2.waitkey(0)

    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('window',grayimg)
    #cv2.waitkey(0)


    facecoordinates=trainedData.detectMultiScale(grayimg)
    #print(facecoordinates)
    for x,y,w,h in facecoordinates:

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('window',img)
    key=cv2.waitKey(1)
    if(key==81 or key==113):
        break
webcam.release()


