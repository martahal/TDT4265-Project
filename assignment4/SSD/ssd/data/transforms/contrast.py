import cv2


class Contrast(object):
    def __call__(self, img, boxes=None, labels=None):
        # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))

        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return img, boxes, labels
