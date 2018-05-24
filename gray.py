import numpy as np
import cv2
import matplotlib.pyplot as plt

#%matplotlib inline

# img_BGR = cv2.imread("S__90128386.jpg")
#img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
#plt.imshow(img_RGB)


img_gray = cv2.imread("S__90128386.jpg",cv2.IMREAD_GRAYSCALE)
#img_gray = cv2.cvtColor(img_BGR, cv2.MREAD_GRAYSCALE)
plt.imshow(img_gray)
plt.colorbar()

plt.hist(img_gray.flatten())

ret, th_img1 = cv2.threshold(img_gray,180,255,cv2.THRESH_BINARY_INV)
plt.imshow(th_img1)

ret, th_img1 = cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY_INV)
plt.imshow(th_img1)
