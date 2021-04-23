import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, shutil

from PIL import Image
from PIL import ImageFilter

def canny():
    img = cv2.imread('datasets/RDD2020_filtered/JPEGImages/Czech_000464.jpg', 0)
    edges = cv2.Canny(img, 150, 250)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def prewitt():
    img = cv2.imread('datasets/RDD2020_filtered/JPEGImages/Japan_007344.jpg', 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    kernelPrewittx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernelPrewitty = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    img_prewittx = cv2.filter2D(img, -1, kernelPrewittx)
    img_prewitty = cv2.filter2D(img, -1, kernelPrewitty)

    img_prewitt = img_prewittx + img_prewitty

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_prewitt, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def higher_contrast(img_path, show_example):
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    img = cv2.imread(img_path, 1)
    if show_example and not img_path:
        img = cv2.imread('datasets/RDD2020_filtered/JPEGImages/Japan_007376.jpg', 1)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    if not show_example:
        return final

    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(final)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def sharpening(img_path, show_example):
    # https://stackoverflow.com/questions/19890054/how-to-sharpen-an-image-in-opencv
    img = cv2.imread(img_path, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    """
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        im = cv2.filter2D(img, -1, kernel)
        cv2.imwrite("Sharpening.png", im)
    """

    aw = cv2.addWeighted(img, 8, cv2.blur(img, (30, 30)), -8, 128)
    if not show_example:
        return aw

    plt.subplot(121), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(aw)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def sharpeningPIL(img_path, show_example):
    img = Image.open('datasets/RDD2020_filtered/JPEGImages/India_001831.jpg')
    sharpen1 = img.filter(ImageFilter.SHARPEN)
    sharpen2 = sharpen1.filter(ImageFilter.SHARPEN)
    sharpen2.show()


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def preprocess_images(should_copy):
    preprocess_method = sharpening
    IN_FOLDER = 'datasets/RDD2020_filtered'
    OUT_FOLDER = 'datasets/RDD2020_filtered_contrast'
    IMAGE_FOLDER_NAME = 'JPEGImages'
    IMAGE_FOLDER = os.path.join(OUT_FOLDER, IMAGE_FOLDER_NAME)

    if should_copy:
        copytree(IN_FOLDER, OUT_FOLDER)

    for filename in os.listdir(IMAGE_FOLDER):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        preprocessed_image = preprocess_method(image_path, False)
        cv2.imwrite(image_path, preprocessed_image)

preprocess_images(True)