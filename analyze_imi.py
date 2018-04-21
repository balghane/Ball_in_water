import cv2
import os
import numpy as np
import os


class Contour:
    def __init__(self):
        return


# these are color filters in HSV
# Hue is 0 to 180
# Saturation and Value are 0 to 255
lower_green = np.array([150, 5, 5])
upper_green = np.array([180, 255, 255])


def analyze_image(frame):
    # change the captured frame from BGR (blue-green-red) to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # filter on color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    n_ero = 1
    ero_size = 5
    kernel_e = np.ones((ero_size, ero_size), np.uint8)
    mask = cv2.erode(mask, kernel_e, iterations=n_ero)

    n_dil = 4
    dil_size = 8
    kernel_d = np.ones((dil_size, dil_size), np.uint8)
    mask = cv2.dilate(mask, kernel_d, iterations=n_dil)

    # cv2.imshow('mask', mask)

    # opencv function that finds shapes
    new_mask, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    newContours = []
    areas = []


    # this is where the code gets weird
    # start by looking at each contour
    for index, c in enumerate(contours):
        # find the arc length (aka circumference) of the contour and that must be bigger than a particular size
        approx = cv2.approxPolyDP(c, 2, True)
        hull = cv2.convexHull(approx)
        hull_area = cv2.contourArea(hull)
        per = cv2.arcLength(hull, True)
        if cv2.contourArea(hull) < 0.01:
            continue
        solidity = cv2.contourArea(approx)/hull_area

        # print(index, per, hull_area, solidity)

        # this is where the filtering happens
        if hull_area < 7000:
            continue

        # this stuff just stores the contours that we like
        contour = Contour()
        areas.append(hull_area)
        M = cv2.moments(hull)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contour.hull_area = hull_area
        contour.contour = hull
        contour.cX = cX
        contour.cY = cY
        contour.radius = int(round(np.sqrt(contour.hull_area / 3.14), 0)) - 5
        newContours.append(contour)

        # print(index, cX, cY, per, cv2.contourArea(approx), cv2.contourArea(hull), solidity)
    # print(len(newContours))

    # figure out how strong the signal is
    newContours.sort(key=lambda x: x.cY, reverse=False)
    ret = newContours[-1]

    # this draws all the good contours
    cv2.drawContours(new_mask, [c.contour for c in newContours], -1, (128, 255, 0), 3)
    cv2.circle(frame, (ret.cX, ret.cY), ret.radius, (0,255,0), thickness=1, lineType=8, shift=0)

    # Display the resulting frame
    # c
    # cv2.imshow('new_mask', new_mask)
    # cv2.imshow('img', frame)
    # cv2.waitKey(0)

    return ret.cX, ret.cY, ret.radius


def main():
    for im_path in os.listdir("./images"):
        print("______________________________")
        print(im_path)
        im = cv2.imread("./images/"+im_path)
        cX, cY, rad = analyze_image(im)
        print(cX, cY, rad)
        cv2.circle(im, (cX, cY), rad, (0,255,0), thickness=1, lineType=8, shift=0)
        cv2.imshow('frame', im)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()