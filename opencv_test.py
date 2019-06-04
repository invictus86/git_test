import cv2

img = cv2.imread("G:/code/git_test/1.jpg", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("Demo")
cv2.startWindowThread()
cv2.imshow("Demo", img)
cv2.waitKey(0)
cv2.destoryAllWindows()
