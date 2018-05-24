import random
import copy
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# %matplotlib inline

# 画像読み込み
# coins_BGR = cv2.imread(IMG_1803.png",cv2.IMREAD_COLOR)
coins = Image.open('IMG_1803.jpg')
coins_BGR =  cv2.imread("IMG_1803.jpg",cv2.IMREAD_COLOR)

# グレースケールで画像読み取り
coins_gray =  cv2.imread("IMG_1803.jpg",0)

# 画素数の確認
# print(np.array(coins).shape)

# オリジナル画像の幅の高さの取得
width, height = coins.size
image_size = height * width

# オリジナル画像と同じサイズのImageオブジェクトを作成する
# coins_RGB = Image.new('RGB', (width, height))

# coins_pixels = []
# for y in range(height):
#   for x in range(width):
#     # getpixel((x,y))で左からx番目,上からy番目のピクセルの色を取得し、img_pixelsに追加する
#     coins_pixels.append(coins.getpixel((x,y)))
# # あとで計算しやすいようにnumpyのarrayに変換しておく
# coins_pixels = np.array(coins_pixels)

# coinsのgray化
# coins_gray = coins.convert('L')

# coinsのgray化画像確認
# coins_gray.save('sample-gray.png')

#coins_grayを配列化
coins_gray_array = np.array(coins_gray)

# coins_gray_arrayの確認
# print(coins_gray_array[0,0])

# coins_gray_arrayのタイプ確認
# print(type(coins_gray_array[0,0]))

# 2値化用、新規配列の作成
coins_binary = np.zeros([coins_gray_array.shape[0],coins_gray_array.shape[1]])

ret, coins_binary_2 = cv2.threshold(coins_gray, 20, 255, cv2.THRESH_BINARY)

#2値化用、新規配列の確認
# print(coins_binary[0,0])

# sample_array = np.random.rand(100)
# plt.hist(sample_array, bins='auto')

# ヒストグラムの生成
# hist = np.histogram(coins_gray_array, bins=1000)
# plt.hist(coins_gray_array, bins=1000)
# plt.show()

#coins_grayの閾値160で2値化
coins_binary[coins_gray_array<120] = 0
coins_binary[coins_gray_array>=120] = 255

# # 2値化画像の確認
# plt.subplot(1,2,1)
# plt.imshow(coins_gray, cmap = 'gray')
# plt.title('Input Image')
# plt.subplot(1,2,2)
# plt.imshow(coins_binary, cmap = 'gray')
# plt.title('Result Image')
# plt.show()
#

# # # for i in range(255):
# # #     if hist[count]>histmax:
# #
# # #
# # histmax = np.amax(hist)


# しきい値指定によるフィルタリング
retval, coins_threshold_filter = cv2.threshold(coins_gray_array, 50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 白黒の反転
coins_color_flip = cv2.bitwise_not(coins_threshold_filter)

# 再度フィルタリング
retval, coins_filter_again = cv2.threshold(coins_color_flip, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 輪郭を抽出
coins_contours, contours, hierarchy = cv2.findContours(coins_filter_again, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# この時点での状態をデバッグ出力
dst = cv2.drawContours(coins_BGR, contours, -1, (0, 0, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite('debug_1.png', dst)

for i, contour in enumerate(contours):
    # 小さな領域の場合は間引く
    area = cv2.contourArea(contour)
    if area < 500:
        continue

    # 画像全体を占める領域は除外する
    if image_size * 0.99 < area:
        continue

    # 外接矩形を取得
    x,y,w,h = cv2.boundingRect(contour)
    dst = cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)

# 結果を保存
cv2.imwrite('result.png', dst)


#labeiingの実験
nLabels, labelImage = cv2.connectedComponents(coins_binary_2)

colors = []
for i in range(1, nLabels + 1):
    colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

for y in range(0, height):
    for x in range(0, width):
        if labelImage[y, x] > 0:
            dst[y, x] = colors[labelImage[y, x]]
        else:
            dst[y, x] = [0, 0, 0]

cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Source", coins_BGR)
cv2.namedWindow("Connected Components", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Connected Components", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
