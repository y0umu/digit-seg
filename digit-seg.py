#!/usr/bin/python3
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from itertools import takewhile

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
    # img = cv2.pyrUp(img)
    # img = cv2.pyrUp(img)
    kernel = np.ones((3,3), np.uint8)
    img_openning = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # img_openning = cv2.pyrDown(img_openning)
    # img_openning = cv2.pyrDown(img_openning)
    return img_openning

# is the background black or white?
def blacken_bg(img):
    hist = cv2.calcHist([img], [0], None, [2], [0,256])
    hist = hist.ravel()
    if hist[0] < hist[1]: # background is white, reverting it
        img = 255 - img
    return img


def get_border_left(projection_vert, prev_border_right, th_x_val):
    '''
    used in function digit_seg()
    自prev_border_right开始寻找下一个左边界
    '''
    n = prev_border_right + 1
    border_left = None # 注意！需要外部代码判断是不是返回了None
    while n < projection_vert.size:
        if projection_vert[n] > th_x_val:
            border_left = n
            break
        n+=1
    return border_left

def get_border_right(projection_vert, border_left, th_x_val):
    '''
    used in function digit_seg()
    返回左边界为projection_horz的patch的右边界
    '''
    n = border_left
    border_right = projection_vert.size  # 初始化为图像最右边，以备意外
    while n < projection_vert.size:
        if projection_vert[n] < th_x_val:
            border_right = n
            break
        n+=1
    return border_right

def get_height(projection_horz, th_y_val):
    '''
    used in function digit_seg()
    寻找数字的高度范围
    '''
    n = 0
    height_up, height_down = 0, projection_horz.size  # note that height_up < height_down ! (up and down are with respect to the image)
    while n < projection_horz.size:
        if projection_horz[n] > th_y_val:
            height_up = n
            break
        n+=1

    n+=1
    while n < projection_horz.size:
        if projection_horz[n] < th_y_val:
            height_down = n
            break
        n+=1
    return height_up, height_down

def digit_seg(img, th_x_factor=0.10, th_y_factor=0.15, want_plt=False):
    '''
    根据竖直和水平投影的信息分割出数字。返回包含所有分割结果图像的borders（横坐标集合）

    传入
    img: 二值图像
    want_plt：如传入True，则画图
    th_x_factor, th_y_factor: 阈值比率。这个参数决定阈值电压：th_x_val = maximum * th_factor
    
    返回
    borders: 返回值。list
    heights: 返回值。tuple
    '''

    imgs_segmented = []
    # project and plot
    
    projection_horz = np.sum(img, axis=1) / 255 # 沿着水平方向投影
    mx_y = np.max(projection_horz)
    th_y_val = mx_y * th_y_factor
    heights = get_height(projection_horz, th_y_val)
    img_roi = img[heights[0]:heights[1],:]

    projection_vert = np.sum(img_roi, axis=0) / 255 # 沿着竖直方向投影
    mx_x = np.max(projection_vert)
    th_x_val = mx_x * th_x_factor
    
    if want_plt:
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(img, cmap="gray"); plt.title("digit_seg() input")
        plt.subplot(2,2,2)
        plt.plot(projection_horz, np.arange(img.shape[0], 0, -1))
        plt.plot(th_y_val*np.ones_like(projection_horz), np.arange(img.shape[0], 0, -1), color="r")
        plt.title("Horizontal projection")
        # plt.yticks(np.arange(img.shape[0], 0, -1))
        plt.subplot(2,2,3)
        plt.plot(np.arange(0, img.shape[1]), projection_vert)
        plt.plot(np.arange(0, img.shape[1]), th_x_val*np.ones_like(projection_vert), color="r")
        plt.title("Vertical projection")
        plt.savefig("output\\projection.png")

    # find the borders 
    borders = []  # e.g. [(2,15), (17,29), (34, 47)]
    n = 0
    border_left = 0
    border_right = -1
    while n < projection_vert.size:
        border_left = get_border_left(projection_vert, border_right, th_x_val)
        if border_left is None:
            break
        border_right = get_border_right(projection_vert, border_left, th_x_val)
        borders.append((border_left, border_right))
    print("vertical borders are ", borders)
    print("get_height: height range {0}".format(heights))
    return borders, heights

def draw_bounding_boxes(img, borders, heights):
    '''
    在img上画框
    返回img_boxed
    '''
    img_boxed = img.copy()
    for i in borders:
        cv2.rectangle(img_boxed, (i[0],heights[0]), (i[1],heights[1]), (255,255,255), 1)
    return img_boxed

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
    img_morph = mkmorph(img_th)
    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(img_gray, cmap="gray")
    plt.xticks([]); plt.yticks([]); plt.title("Grayscale image")
    plt.subplot(3,1,2)
    plt.imshow(img_th, cmap="gray")
    plt.xticks([]); plt.yticks([]); plt.title("Thresholded image")
    plt.subplot(3,1,3)
    plt.imshow(img_morph, cmap="gray")
    plt.xticks([]); plt.yticks([]); plt.title("Image applied with openning")
    plt.savefig("output\\00_intermediates.png")
    
    the_img = blacken_bg(img_morph)
    borders, heights = digit_seg(the_img, want_plt=True)
    imgs_segmented = []
    for i in borders:
        imgs_segmented.append(img[heights[0]:heights[1]+1, i[0]:i[1]+1])
    n_segs = len(imgs_segmented)

    _img_boxed = draw_bounding_boxes(img, borders, heights)
    img_boxed = cv2.cvtColor(_img_boxed, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_boxed)
    plt.savefig("output\\01_segmented_image_boxed.png")
    plt.xticks([]); plt.yticks([]); plt.title("Original image with bounding boxes")
    plt.figure()
    for (num, _im) in enumerate(imgs_segmented):
        im = cv2.cvtColor(_im, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n_segs, num+1)
        plt.imshow(im)
        plt.xticks([]); plt.yticks([]); plt.title("No."+str(num+1))
    plt.savefig("output\\02_segmented_image_subplot.png")
    plt.show()
    print("done")


