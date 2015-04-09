import numpy as np
import cv2
import pdb
import time

cap = cv2.VideoCapture('Highway5.mp4')
# cap = cv2.VideoCapture(0)

fps = 30
capSize = (960,540)
fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
vidWrite = cv2.VideoWriter()
success = vidWrite.open('opticFlow.mp4',fourcc,fps,capSize,True)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 15,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


# Take first frame and find corners in it
p0 = None

# pdb.set_trace()

while p0 is None:
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)



# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
weighted = np.float32(old_frame)


while(1):

    Flag = False;
    ret,frame = cap.read()

    if ret == False:
        break;
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    while p0 is None:    
        ret, old_frame = cap.read()
        frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)


    
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.circle(frame,(a,b),5,(0,255,0),-1)
    

    img = cv2.add(frame,mask)

    # cv2.imshow('mog',old_gray)
    # cv2.imshow('hybrid',img)
    vidWrite.write(img)
    # print frame_gray.shape
    
    k = cv2.waitKey(1)
    if k == 27:
        break
   
    # Now update the previous frame and previous points
    old_gray = frame_gray
    p0 = good_new.reshape(-1,1,2)
    
    
   
vidWrite.release()
cv2.destroyAllWindows()
cap.release()