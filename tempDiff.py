import cv2

def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.absdiff(d2,d1)
  #return cv2.bitwise_and(d1, d2)

fps = 30
capSize = (960,540)
fourcc = cv2.cv.CV_FOURCC('m','p','4','v')
vidWrite = cv2.VideoWriter()
success = vidWrite.open('tempDiff.mp4',fourcc,fps,capSize,True)

cam = cv2.VideoCapture('Highway5.mp4')

winName = "Movement Indicator"
cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)

# Read three images first:
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

while True:
  vidWrite.write(cv2.cvtColor(diffImg(t_minus, t, t_plus), cv2.COLOR_GRAY2RGB) )
  # Read next image
  t_minus = t
  t = t_plus
  t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

  key = cv2.waitKey(10)
  if key == 27:
    cv2.destroyWindow(winName)
    break


vidWrite.release()
