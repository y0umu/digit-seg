#!/usr/bin/python3
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

cmd_parser = argparse.ArgumentParser(prog="digit-srg.py",
                                     description="从图像中分割出数字")
cmd_parser.add_argument("image", help="输入图像")
cmd_args = cmd_parser.parse_args()

g_input_img_name = cmd_args.image

img = cv2.imread(g_input_img_name)
if (img is None):
    raise ValueError("cannot read " + g_input_img_name)

# grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("img_gray", cv2.WINDOW_NORMAL)
cv2.imshow("img_gray", img_gray)
cv2.waitKey()

# thresholding
th_ret, img_th= cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("img_th", img_th)
cv2.waitKey()

# morphing
kernel = np.ones((3,3), np.uint8)
img_openning = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)
cv2.imshow("img_openning", img_openning)
cv2.waitKey()

# is the background black or white?
hist = cv2.calcHist([img_openning], [0], None, [2], [0,256])
hist = hist.ravel()
if hist[0] < hist[1]:
    print("background is white")
else:
    print("background is black")