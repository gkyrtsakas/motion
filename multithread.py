import cv2
import numpy as np
import threading
import Queue
import sys
import pdb
import time

cap = cv2.VideoCapture('Highway3.mp4')
frameQueue = Queue.Queue(maxsize=0)
lock = threading.Lock()

def getFrame(q):
	ret,frame = cap.read()
	while ret == True:
		# with lock:
			# 	capturing = True
		q.put(frame)
		# print q.qsize()
		ret,frame = cap.read()
	# with lock:
	# 	capturing = False
	cap.release()
	print "capture done"


getFrameThread = threading.Thread(target=getFrame, args=(frameQueue,))
getFrameThread.setDaemon(True)
getFrameThread.start()
time.sleep(1)
	# pdb.set_trace()

while (frameQueue.empty()!=True):
	# print frameQueue.qsize()
	
	output = frameQueue.get()

	cv2.imshow('frame',output)
    	# print frame_gray.shape
    
   	k = cv2.waitKey(33)
   	if k == 27:
   		break
frameQueue.join()
cv2.destroyAllWindows()
