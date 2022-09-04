"""
Steps to completion of the Project:
1. Detect the edges
2. Get the biggest contour (the document).
3. Get the points of the contour.
4. Warp the contour.
"""

import cv2
import numpy as np

#################################################################
img_width = 300
img_height = 200
warped_img_width = 200
warped_img_height = 300
min_area = 2000
################################################################

cap = cv2.VideoCapture(1)

def pre_processing(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5,5), 1)
    canny_img = cv2.Canny(blur_img, 200, 200)
    # kernel = np.ones((5,5))
    # dilate_img = cv2.dilate(canny_img, kernel, iterations=2)
    # eroded_img = cv2.erode(dilate_img, kernel, iterations=1)
    return canny_img

def get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_area = 0
    biggest_cnt = np.array([])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            #cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            corners = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if area > largest_area and len(corners) == 4:
                largest_area = area
                biggest_cnt = corners
    cv2.drawContours(img_contour, biggest_cnt, -1, (255, 0, 0), 20)
    return biggest_cnt

def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)

    sum = points.sum(1)
    new_points[0] = points[np.argmin(sum)]
    new_points[3] = points[np.argmax(sum)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points

def get_warp(img, biggest_cnt):
    biggest_cnt = reorder(biggest_cnt)
    input_pts = np.float32(biggest_cnt)
    output_pts = np.float32([[0, 0], [warped_img_width, 0], [0, warped_img_height], [warped_img_width, warped_img_height]])
    matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
    output_img = cv2.warpPerspective(img, matrix, (warped_img_width, warped_img_height))

    cropped = output_img[5: output_img.shape[0]-5, 5: output_img.shape[1]-5]
    cropped = cv2.resize(cropped, (warped_img_width, warped_img_height))

    return cropped

while True:
    success, img = cap.read()
    img = cv2.resize(img, (img_width, img_height))
    img_contour = img.copy()
    threshold_img = pre_processing(img)
    threshold_img_RGB = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)
    biggest_cnt = get_contours(threshold_img)
    print(biggest_cnt)
    warped_img = np.zeros((warped_img_height, warped_img_width))

    if len(biggest_cnt) != 0:
        warped_img = get_warp(img, biggest_cnt)

    h_stack = np.hstack((img, threshold_img_RGB, img_contour))
    cv2.imshow("Stacked images", h_stack)
    cv2.imshow("Result", warped_img)

    if cv2.waitKey(1) & 0xFF == 'q':
        break
