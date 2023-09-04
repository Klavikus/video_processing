import cv2
import numpy as np
import matplotlib.image as impl

import os


class ImageFilter:
    def __init__(self, config):
        self.config = config
        self.filter_data = {}

    def process(self, image):
        raise NotImplementedError('Subclasses must implement process method')


class SaveAsPng(ImageFilter):
    def __init__(self, config, save_path, save_name):
        super().__init__(config)
        self.save_path = save_path
        self.save_name = save_name
        self.save_counter = 0
        create_directory_if_not_exists(save_path)

    def process(self, image, grayscale=False):
        save_image(image, f'{self.save_path}/{self.save_name}_{self.save_counter}.png', 'png')
        self.save_counter += 1
        return image


class BinaryzationFilter(ImageFilter):
    def process(self, image):
        if self.config['USE_ADAPTIVE']:
            (T, img_binary) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            img_binary = binary_segmentation(image,
                                             self.config['BINARY_THRESHOLD_MIN'],
                                             self.config['BINARY_THRESHOLD_MAX'],
                                             255)

        img_binary = set_clean_boundary(img_binary, thickness=0)
        return img_binary


class MorphologicalFilter(ImageFilter):

    def process(self, image):
        ops_for_img = self.config['MORPH_OPERATIONS']

        img_morph = cv_op(image, ops_for_img)
        img_morph = set_clean_boundary(img_morph)
        # img_morph = cv2.GaussianBlur(img_morph,
        #                              (self.config['GAUSSIAN_K'], self.config['GAUSSIAN_K']),
        #                              0)
        return img_morph


class ContourDetectionFilter(ImageFilter):
    def process(self, image):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        approx_contours = []

        contours_image = np.zeros(image.shape[:2])

        for cnt in contours:
            epsilon = self.config['CONTOUR_EPSILON'] * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx_contours.append(approx)

        valid_contours = []
        new_hierarchy = []
        mask_holes = np.zeros(image.shape[:2], np.uint8)
        for i, contour in enumerate(approx_contours):
            if len(contour) > self.config['CONTOUR_MIN_LENGTH'] and cv2.contourArea(contour) > \
                    self.config['CONTOUR_MIN_AREA']:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area
                if hierarchy[0][i][3] != -1:
                    cv2.drawContours(mask_holes, [contour], -1, 255, -1)
                if hierarchy[0][i][3] != -1 or solidity <= self.config['CONTOUR_MIN_SOLIDITY']:
                    continue
                contour_moments = cv2.moments(contour)
                center_x = int(contour_moments["m10"] / contour_moments["m00"])
                center_y = int(contour_moments["m01"] / contour_moments["m00"])
                if not check_overlapping((center_x, center_y), (0, 0), (2 * 37, 2 * 16)):
                    valid_contours.append(contour)
                    new_hierarchy.append(hierarchy[0][i])

        contours_image = cv2.drawContours(contours_image, valid_contours, -1, (255, 255, 255), 3)
        self.filter_data['contours'] = valid_contours
        self.filter_data['mask_holes'] = mask_holes
        return contours_image


class ConverterBGR2GRAY(ImageFilter):
    def process(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


class CropFilter(ImageFilter):
    def __init__(self, config, x_slice=(0, 0), y_slice=(0, 0)):
        super().__init__(config)
        self.x_slice = x_slice
        self.y_slice = y_slice

    def process(self, image):
        start_x = self.x_slice[0]
        end_x = image.shape[:2][0] if self.x_slice[1] == 0 else self.x_slice[1]
        start_y = self.y_slice[0]
        end_y = image.shape[:2][1] if self.y_slice[1] == 0 else self.y_slice[1]

        self.filter_data['start_shape'] = image.shape[:2]
        self.filter_data['end_shape'] = image[start_x:end_x, start_y:end_y].shape[:2]

        return image[start_x:end_x, start_y:end_y]


class ReplaceFilter(ImageFilter):
    def __init__(self, config, x_slice=(0, 0), y_slice=(0, 0), color=0):
        super().__init__(config)
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.color = color

    def process(self, image):
        for i in range(self.x_slice[0], self.x_slice[1]):
            for j in range(self.y_slice[0], self.y_slice[1]):
                image[i][j] = self.color
        return image


class DenoiseFilter(ImageFilter):
    def process(self, image):
        scale_factor = 1
        image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        img_uint8 = np.uint8(image)
        #
        # if self.config['USE_NLM']:
        #     nlm = cv2.fastNlMeansDenoising(img_uint8,
        #                                    h=self.config['NLM_H'],
        #                                    templateWindowSize=self.config['NLM_TEMPLATE_WINDOW_SIZE'],
        #                                    searchWindowSize=self.config['NLM_TEMPLATE_WINDOW_SIZE'])
        # else:
        #     nlm = img_uint8
        #
        # img_bilateral = cv2.bilateralFilter(nlm,
        #                                     self.config['BILATERAL_D'],
        #                                     self.config['BILATERAL_SIGMA_COLOR'],
        #                                     self.config['BILATERAL_SIGMA_SPACE'])
        #
        # img_median = cv2.medianBlur(img_bilateral, self.config['MEDIAN_K'])
        #
        # denoised_image = cv2.GaussianBlur(img_median,
        #                                   (self.config['GAUSSIAN_K'], self.config['GAUSSIAN_K']),
        #                                   0)
        denoised_image = img_uint8
        return denoised_image


class NeuroBinary(ImageFilter):
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model

    def process(self, image):
        initial_shape = image.shape
        resized_image = cv2.resize(image, (512, 512))
        processed_image = np.expand_dims(resized_image, axis=2)
        processed_image = np.repeat(processed_image, 3, axis=2)
        processed_image = np.expand_dims(processed_image, axis=0)
        predicted_image = self.model.predict(processed_image)[0]
        initial_shape_predicted_image = cv2.resize(predicted_image, (initial_shape[1], initial_shape[0]))
        image_uint8 = (initial_shape_predicted_image * 255).astype(np.uint8)
        return image_uint8


class TrackingModule:
    def __init__(self, centroid_tracker):
        self.centroid_tracker = centroid_tracker

    def process(self, image, contours, mask_holes):
        bounding_rectangles = []
        for contour in contours:
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            holes = mask_holes > 0
            cnt_mask = mask - mask_holes
            cnt_mask[holes] = 0
            count = cv2.countNonZero(cnt_mask)
            x, y, w, h = cv2.boundingRect(contour)
            area = count
            perimeter = cv2.arcLength(contour, True)
            bounding_rectangles.append((x, y, x + w, y + h, area, perimeter, contour))

        self.centroid_tracker.update(bounding_rectangles)
        return image


def binary_segmentation(img, min_value, max_value, color=255):
    binary_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    data_segment = (img >= min_value) & (img <= max_value)
    binary_image[data_segment] = color
    return binary_image


def set_clean_boundary(img, thickness=4):
    img[:thickness, :img.shape[1]] = 0
    img[img.shape[0] - thickness:img.shape[0], :img.shape[1]] = 0
    img[:img.shape[0], :thickness] = 0
    img[:img.shape[0], img.shape[1] - thickness:img.shape[1]] = 0
    return img


def check_overlapping(input_point, start_point, end_point):
    return start_point[0] <= input_point[0] <= end_point[0] and start_point[1] <= input_point[1] <= end_point[1]


def save_image(image, save_path, file_format):
    if file_format == "png":
        impl.imsave(save_path, image, format="png", cmap="gray")
    elif file_format == "jpg":
        impl.imsave(save_path, image, format="jpg", cmap="gray")
    elif file_format == "tiff":
        impl.imsave(save_path, image, format="tiff", cmap="gray")
    else:
        raise ValueError("Unsupported file format")


def create_directory_if_not_exists(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


cv2_morphology_operation = {
    'erode': lambda x, y: cv2.erode(x, np.ones(y[0], np.uint8), iterations=1),
    'dilate': lambda x, y: cv2.dilate(x, np.ones(y[0], np.uint8), iterations=1),
    'open': lambda x, y: cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones(y[0], np.uint8)),
    'close': lambda x, y: cv2.morphologyEx(x, cv2.MORPH_CLOSE, np.ones(y[0], np.uint8)),
}


def cv_op(img, operation_list):
    for operation in operation_list:
        splitted = operation.split('_')
        key = splitted[0]
        kern_size = int(splitted[1])
        iterations = int(splitted[2])

        for i in range(iterations):
            img = cv2_morphology_operation[key](img, ((kern_size, kern_size), iterations))
    return img
