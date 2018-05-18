#!/usr/bin/python3
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from itertools import takewhile

# grayscale
def mkgray(img):
    '''
    灰度化，返回灰度图像
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# thresholding
def mkthresh(img):
    '''
    用大津法(OTSU)进行阈值、二值化
    '''
    th_ret, img_th= cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_th

# morphing
def mkmorph(img):
    '''
    形态学处理孤立小点等
    '''
    kernel = np.ones((3,3), np.uint8)
    img_openning = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img_openning

# is the background black or white?
def blacken_bg(img):
    hist = cv2.calcHist([img], [0], None, [2], [0,256])
    hist = hist.ravel()
    if hist[0] < hist[1]: # background is white, reverting it
        img = 255 - img
    return img

def get_border_left(projection_vert, prev_border_right, th_val):
    '''
    used in function digit_seg()
    自prev_border_right开始寻找下一个左边界
    '''
    n = prev_border_right + 1
    border_left = None # 注意！需要外部代码判断是不是返回了None
    while n < projection_vert.size:
        if projection_vert[n] > th_val:
            border_left = n
            break
        n+=1
    return border_left

def get_border_right(projection_vert, border_left, th_val):
    '''
    used in function digit_seg()
    返回左边界为projection_horz的patch的右边界
    '''
    n = border_left
    border_right = projection_vert.size  # 初始化为图像最右边，以备意外
    while n < projection_vert.size:
        if projection_vert[n] < th_val:
            border_right = n
            break
        n+=1
    return border_right

def digit_seg(img, th_factor=0.20, want_plt=False):
    '''
    根据竖直和水平投影的信息分割出数字。返回包含所有分割结果图像的borders

    传入
    img: 二值图像
    want_plt：如传入True，则画图
    th_factor: 阈值比率。这个参数决定阈值电压：th_val = mx * th_factor
    
    返回
    borders: 返回值。list
    '''

    imgs_segmented = []
    # project and plot
    projection_vert = np.sum(img, axis=0) / 255 # 沿着竖直方向投影
    projection_horz = np.sum(img, axis=1) / 255 # 沿着水平方向投影
    mx = np.max(projection_vert)
    th_val = mx * th_factor
    if want_plt:
        plt.subplot(2,2,1)
        plt.imshow(img, cmap="gray"); plt.title("digit_seg() input")
        plt.subplot(2,2,2)
        plt.plot(projection_horz, np.arange(0, img.shape[0])); plt.title("horz projection")
        plt.subplot(2,2,3)
        plt.plot(np.arange(0, img.shape[1]), projection_vert)
        plt.plot(np.arange(0, img.shape[1]), th_val*np.ones_like(projection_vert), color="r")
        plt.title("vert projection")

    # find the borders
    borders = []  # e.g. [(2,15), (17,29), (34, 47)]
    n = 0
    border_left = 0
    border_right = -1
    while n < projection_vert.size:
        border_left = get_border_left(projection_vert, border_right, th_val)
        if border_left is None:
            break
        border_right = get_border_right(projection_vert, border_left, th_val)
        borders.append((border_left, border_right))
    print("borders are ", borders)
    return borders

#################################################################
if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(prog="digit-srg.py",
                                        description="从图像中分割出数字")
    cmd_parser.add_argument("image", help="输入图像")
    cmd_args = cmd_parser.parse_args()

    g_input_img_name = cmd_args.image

    img = cv2.imread(g_input_img_name)
    if (img is None):
        raise ValueError("cannot read " + g_input_img_name)

    img_gray = mkgray(img)
    img_th = mkthresh(img_gray)
    # img_morph = mkmorph(img_th)
    # the_img = blacken_bg(img_morph)
    the_img = blacken_bg(img_th)
    cv2.namedWindow("img_gray", cv2.WINDOW_NORMAL)
    cv2.imshow("img_gray", img_gray)
    
    borders = digit_seg(the_img, want_plt=True)
    imgs_segmented = []
    for i in borders:
        imgs_segmented.append(img[:, i[0]:i[1]])
    for (num, im) in enumerate(imgs_segmented):
        cv2.imshow(str(num), im)
    plt.show()
    cv2.waitKey()
    print("done")


