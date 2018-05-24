import random
import copy
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# %matplotlib inline



#############################################
# ハフ変換
def houghCircles(src):
    #　3*3のメジアンフィルタ処理
    blur = cv2.medianBlur(src, 5)

    # ハフ変換による円の中心座標と半径の抽出
    # 全硬貨を検出対象
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=30, maxRadius=60)

    area = np.zeros(len(circles[0]))

    # 検出値を少数第一位で四捨五入し、16bit符号なし整数に変換
    circles = np.uint16(np.around(circles))
    # print(circles)

    # if circles
    # circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=15)

    i = 0
    for (x, y, r) in circles[0]:
        i += 1
        cv2.circle(src, (x, y), r, (0, 255, 0), 2)
        cv2.circle(src, (x, y), 2, (0, 0, 255), 3)
        area[i-1] = r * r * 3.14
        print(area)

    cv2.imshow('detected circles',src)
#############################################

# ##################################
# # 膨張処理
# def dilate(src, ksize=3):
#     # 入力画像のサイズを取得
#     h, w = src.shape
#     # 入力画像をコピーして出力画像用配列を生成
#     dst = src.copy()
#     # 注目領域の幅
#     d = int((ksize-1)/2)
#
#     for y in range(0, h):
#         for x in range(0, w):
#             # 近傍に白い画素が1つでもあれば、注目画素を白色に塗り替える
#             roi = src[y-d:y+d+1, x-d:x+d+1]
#             if np.count_nonzero(roi) > 0:
#                 dst[y][x] = 255
#
#     return dst
# ##################################
#
#
# ##################################
# # 収縮処理
# def erode(src, ksize=3):
#     # 入力画像のサイズを取得
#     h, w = src.shape
#     # 入力画像をコピーして出力画像用配列を生成
#     dst = src.copy()
#     # 注目領域の幅
#     d = int((ksize-1)/2)
#
#     for y in range(0, h):
#         for x in range(0, w):
#             # 近傍に黒い画素が1つでもあれば、注目画素を黒色に塗り替える
#             roi = src[y-d:y+d+1, x-d:x+d+1]
#             if roi.size - np.count_nonzero(roi) > 0:
#                 dst[y][x] = 0
#
#     return dst
# ##################################
#
#
# ##################################
# # ローパスフィルタ処理メソッド
# def lowpass_filter(src, a = 0.5):
#     # 高速フーリエ変換(2次元)
#     src = np.fft.fft2(src)
#
#     # 画像サイズ
#     h, w = src.shape
#
#     # 画像の中心座標
#     cy, cx =  int(h/2), int(w/2)
#
#     # フィルタのサイズ(矩形の高さと幅)
#     rh, rw = int(a*cy), int(a*cx)
#
#     # 第1象限と第3象限、第1象限と第4象限を入れ替え
#     fsrc =  np.fft.fftshift(src)
#
#     # 入力画像と同じサイズで値0の配列を生成
#     fdst = np.zeros(src.shape, dtype=complex)
#
#     # 中心部分の値だけ代入（中心部分以外は0のまま）
#     fdst[cy-rh:cy+rh, cx-rw:cx+rw] = fsrc[cy-rh:cy+rh, cx-rw:cx+rw]
#
#     # 第1象限と第3象限、第1象限と第4象限を入れ替え(元に戻す)
#     fdst =  np.fft.fftshift(fdst)
#
#     # 高速逆フーリエ変換
#     dst = np.fft.ifft2(fdst)
#
#     # 実部の値のみを取り出し、符号なし整数型に変換して返す
#     return  np.uint8(dst.real)
# ####################################



# 画像読み込み
# coins_BGR = cv2.imread(IMG_1803.png",cv2.IMREAD_COLOR)
coins = Image.open('IMG_1822.jpg')
coins_BGR =  cv2.imread("IMG_1822.jpg",cv2.IMREAD_COLOR)

# グレースケールで画像読み取り
coins_gray =  cv2.imread("IMG_1822.jpg",0)

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

#coinsのgray化画像確認
# coins_gray.save('sample-gray.png')
# cv2.imwrite("gray.png",coins_gray)

# # ローパスフィルタ処理の呼び出し
# coins_gray_LP = lowpass_filter(coins_gray, 0.3)

# # 処理結果を出力
# cv2.imwrite("output.png", coins_gray_LP)


#coins_grayを配列化
coins_gray_array = np.array(coins_gray)

## coins_gray_arrayの確認
# print(coins_gray_array[0,0])

# coins_gray_arrayのタイプ確認
# print(type(coins_gray_array[0,0]))


#########################################
# 2値化処理を行うプログラム

# 2値化用、新規配列の作成
coins_binary = np.zeros([coins_gray_array.shape[0],coins_gray_array.shape[1]])

# 2値化を直接思考
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
coins_binary[coins_gray_array<140] = 0
coins_binary[coins_gray_array>=140] = 255

# # 2値化画像の確認
# plt.subplot(1,2,1)
# plt.imshow(coins_gray, cmap = 'gray')
# plt.title('Input Image')
# plt.subplot(1,2,2)
# plt.imshow(coins_binary, cmap = 'gray')
# plt.title('Result Image')
# plt.show()
#########################################


houghCircles(coins_gray)

# #############################################
# # 面積値から合計金額を算出
#
# one_yen = 0
# five_yen = 0
# ten_yen = 0
# five_ten_yen = 0
# hundred_yen = 0
# five_hundred_yen = 0
#
# for j in range(7):
#     if area[j] < 3800:
#         break
#     elif area[j] < 4000:
#         one_yen += 1
#     elif area[j] < 5000:
#         five_ten_yen += 1
#     elif area[j] < 5800:
#         five_yen += 1
#     elif area[j] < 6000:
#         ten_yen += 1
#     elif area[j] < 7000:
#         hundred_yen += 1
#     elif area[j] < 8500:
#         five_hundred_yen += 1
#
#
# sum = one_yen + five_yen*5 + ten_yen*10 + five_ten_yen*50 + hundred_yen*100 + five_hundred_yen*500
# print(sum)
# #############################################


# # for i in range(255):
# #     if hist[count]>histmax:
#
# histmax = np.amax(hist)

# # 膨張・収縮処理(方法1)
# dilate_img = dilate(coins_binary, ksize=6)
# erode_img = erode(dilate_img, ksize=6)
#
# # 膨張収縮処理の結果を出力
# cv2.imwrite("dilate.png", dilate_img)
# cv2.imwrite("erode.png", erode_img)


# for i in range(0, len(circles)):
# 	area = cv2.contourArea(circles[i])
#
# 	if area < 2826 or 5024 < area:
		# continue

    # if len(circles[i]) > 0:
    #     rect = circle[i]
    #     x,y,w,h = v2.boundingRect(rect)
    #     cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # if(x>y):
	# 	yoko_count = yoko_count + 1
	# elif(y>x):
	# 	tate_count = tate_count + 1
    #
    #
	# detect_count = detect_count + 1


# #############################################
# #　外接矩形を出力するためのプログラム
#
# # しきい値指定によるフィルタリング
# retval, coins_threshold_filter = cv2.threshold(coins_gray_array, 50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
# # 白黒の反転
# coins_color_flip = cv2.bitwise_not(coins_threshold_filter)
#
# # 再度フィルタリング
# retval, coins_filter_again = cv2.threshold(coins_color_flip, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
# # 輪郭を抽出
# coins_contours, contours, hierarchy = cv2.findContours(coins_filter_again, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # この時点での状態をデバッグ出力
# dst = cv2.drawContours(coins_BGR, contours, -1, (0, 0, 255, 255), 2, cv2.LINE_AA)
# cv2.imwrite('debug_1.png', dst)
#
# for i, contour in enumerate(contours):
#     # 小さな領域の場合は間引く
#     area = cv2.contourArea(contour)
#     if area < 500:
#         continue
#
#     # 画像全体を占める領域は除外する
#     if image_size * 0.99 < area:
#         continue
#
#     # 外接矩形を取得
#     x,y,w,h = cv2.boundingRect(contour)
#     dst = cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
#
# # 外接矩形の出力結果を保存
# cv2.imwrite('result.png', dst)
# ###############################################


################################################
# #labeiingの実装
# nLabels, labelImage = cv2.connectedComponents(coins_binary_2)
#
# colors = []
# for i in range(1, nLabels + 1):
#     colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))
#
# for y in range(0, height):
#     for x in range(0, width):
#         if labelImage[y, x] > 0:
#             dst[y, x] = colors[labelImage[y, x]]
#         else:
#             dst[y, x] = [0, 0, 0]
#
# cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("Source", coins_BGR)
# cv2.namedWindow("Connected Components", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("Connected Components", dst)
################################################


cv2.waitKey(0)
cv2.destroyAllWindows()
