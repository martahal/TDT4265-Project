import cv2
import numpy as np
from matplotlib import pyplot as plt


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

def higher_contrast():
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    img = cv2.imread('datasets/RDD2020_filtered/JPEGImages/Japan_007376.jpg', 1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(final)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

higher_contrast()


def sharpening():
    # https://stackoverflow.com/questions/19890054/how-to-sharpen-an-image-in-opencv
    img = cv2.imread('datasets/RDD2020_filtered/JPEGImages/Japan_007344.jpg', 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    im = cv2.filter2D(img, -1, kernel)
    cv2.imwrite("Sharpening.png", im)


    aw = cv2.addWeighted(img, 8, cv2.blur(img, (30, 30)), -8, 128)
    plt.subplot(121), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(im)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()