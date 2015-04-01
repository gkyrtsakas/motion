import numpy as np
import cv2

try:
  cap = cv2.VideoCapture('footage3.mp4')
except:
  print "Could not load video"

fps = 20
capSize = (480,360)
fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
mog = cv2.BackgroundSubtractorMOG()
#mog2 = cv2.BackgroundSubtractorMOG2()
vidWrite = cv2.VideoWriter()
success = vidWrite.open('backSubOut.mp4',fourcc,fps,capSize,True)

while(1):
    ret, frame = cap.read()
    if ret == False:
      break;

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mog_image  = mog.apply(frame)
    #mog2_image  = mog2.apply(frame)

    #mog2_image  = cv2.resize(mog2_image, (0,0), fx=0.5, fy=0.5)
    #mog_image  = cv2.resize(mog_image, (0,0), fx=0.5, fy=0.5)
    #frame   = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)


    cv2.imshow('MOG',mog_image)
    mog_image = cv2.cvtColor(mog_image, cv2.COLOR_GRAY2BGR);
    #print mog_image.shape
    vidWrite.write(mog_image)
    
    #cv2.imshow('MOG2',mog2_image)
    #cv2.imshow('GMG',gmg_image)
    #cv2.imshow('Original',frame)
    k = cv2.waitKey(10)
    if k == 27:
        break

vidWrite.release()
cap.release()
cv2.destroyAllWindows()
