import cv2

trainedData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread("C:/Users/DELL/Downloads/FACE DEC/pic4.jpg")
# cv2.imshow('pic2',img)
# cv2.waitKey(0)


grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('window',grayimg)
# cv2.waitKey(0.2)


facecoordinates = trainedData.detectMultiScale(grayimg)

# print(facecoordinates)
for x, y, w, h in facecoordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('window', img)
cv2.waitKey(0)
