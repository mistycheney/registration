import cv2
if __name__ == "__main__":
     img = cv2.imread('allen0.tif',0)
     cv2.imshow('img',img)
     cv2.waitKey(0)
