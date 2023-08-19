import cv2
import numpy as np
import matplotlib.image as impl
import matplotlib.pyplot as plt

import os


class ImageFilter:
    def __init__(self, config):
        self.config = config
        self.filter_data = {}

    def process(self, image):
        raise NotImplementedError('Subclasses must implement process method.&quot')


class SaveAsPng(ImageFilter):
    def __init__(self, config, save_path, save_name):
        super().__init__(config)
        self.save_path = save_path
        self.save_name = save_name
        self.save_counter = 0
        create_directory_if_not_exists(save_path)

    def process(self, image, grayscale=False):
        save_image(image, f'{self.save_path}/{self.save_name}_{self.save_counter}.png', 'png', grayscale)
        self.save_counter += 1
        return image


class ShowImage(ImageFilter):
    def __init__(self, config, cmap):
        super().__init__(config)
        self.cmap = cmap

    def process(self, image):
        plt.imshow(image, cmap=self.cmap)
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
        # Алгоритм бинаризации изображения
        # Возвращается измененное изображение
        return img_binary


class ContourDetectionFilter(ImageFilter):
    def process(self, image):
        # Алгоритм обнаружения контуров на изображении
        # Возвращается измененное изображение
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        approx_contours = []

        contours_image = np.zeros(image.shape[:2])

        for cnt in contours:
            epsilon = self.config['CONTOUR_EPSILON'] * cv2.arcLength(cnt, True)
            # epsilon = self.config['CONTOUR_EPSILON'] * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx_contours.append(approx)

        contours_image = cv2.drawContours(contours_image, approx_contours, -1, (255, 255, 255), 3)

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
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:-36, :]

        scale_factor = 1
        image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

        dx = 32
        dy = 74
        image[0:dx, 0:dy] = 0

        # image = np.clip(image, 0, 1)

        img_uint8 = np.uint8(image)

        if self.config['USE_NLM']:
            nlm = cv2.fastNlMeansDenoising(img_uint8,
                                           h=self.config['NLM_H'],
                                           templateWindowSize=self.config['NLM_TEMPLATE_WINDOW_SIZE'],
                                           searchWindowSize=self.config['NLM_TEMPLATE_WINDOW_SIZE'])
        else:
            nlm = img_uint8

        img_bilateral = cv2.bilateralFilter(nlm,
                                            self.config['BILATERAL_D'],
                                            self.config['BILATERAL_SIGMA_COLOR'],
                                            self.config['BILATERAL_SIGMA_SPACE'])

        img_median = cv2.medianBlur(img_bilateral, self.config['MEDIAN_K'])

        denoised_image = cv2.GaussianBlur(img_median,
                                          (self.config['GAUSSIAN_K'], self.config['GAUSSIAN_K']),
                                          0)

        return denoised_image


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


def save_image(image, save_path, file_format, grayscale):
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
