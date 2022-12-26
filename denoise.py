import cv2


def calculate_iou(binary_img, ground_truth_img):
    img_bwa = cv2.bitwise_and(binary_img, ground_truth_img)
    img_bwo = cv2.bitwise_or(binary_img, ground_truth_img)
    count_bwa = cv2.countNonZero(img_bwa)
    count_bwo = cv2.countNonZero(img_bwo)
    return count_bwa/count_bwo