import cv2
import os

for im_path in os.listdir("./screenshots"):
    print(im_path)
    im = cv2.imread("./screenshots/"+im_path)
    im = im[285:690, 683:1223]

    cv2.imwrite("./images/"+im_path, im)

    cv2.imshow("im", im)
    # cv2.waitKey(0)
