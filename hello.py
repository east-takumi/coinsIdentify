import numpy as np
import cv2


# 10 x 10 の3レイヤー(BGR)を定義
size = 10, 10, 3

# cv2.fillPolyで赤に埋める
red_img = np.zeros(size, dtype=np.uint8)
contours = np.array( [ [0,0], [0,10], [10, 10], [10,0] ] )
cv2.fillPoly(red_img, pts =[contours], color=(0,0,255))

# cv2.rectangleで青に埋める
blue_img = np.zeros(size, dtype=np.uint8)
cv2.rectangle(blue_img,(0,0),(10,10),(255,0,0),cv2.CV_FILLED)

# np.fillで白に埋める
white_img = np.zeros(size, dtype=np.uint8)
white_img.fill(255)

# np.tileで緑に埋める
green_img = np.tile(np.uint8([0,255,0]), (10,10,1))

# 10 x 10 の単階調で灰色に埋める
gray_img = np.tile(np.uint8([127]), (10,10,1))

# リストの最初から最後までを紫に埋める
# 参考: http://stackoverflow.com/questions/4337902/how-to-fill-opencv-image-with-one-solid-color
purple_img = np.zeros(size, dtype=np.uint8)
purple_img[:] = (255, 0, 255)

# RGBの並びをBGRに変換して黄色に埋める
yellow_img = np.zeros(size, dtype=np.uint8)
rgb_color = (255,255,0);
yellow_img[:] = tuple(reversed(rgb_color))

cv2.namedWindow("yellow image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("yellow image",yellow_img)
cv2.waitKey(0)
