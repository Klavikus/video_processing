import cv2
import numpy as np
from object_tracker import CentroidTracker
from matplotlib import pyplot as plt
from data_processing import get_mean_data, get_inverse_dict
import skvideo
import skvideo.io
from pathlib import Path
import pandas as pd
from settings import Config

from n2v.models import N2V

cv2_morphology_operation = {
    'erode': lambda x, y: cv2.erode(x, np.ones(y[0], np.uint8), iterations=1),
    'dilate': lambda x, y: cv2.dilate(x, np.ones(y[0], np.uint8), iterations=1),
    'open': lambda x, y: cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones(y[0], np.uint8)),
    'close': lambda x, y: cv2.morphologyEx(x, cv2.MORPH_CLOSE, np.ones(y[0], np.uint8)),
}


def cv_op(img, operation_list):
    for operation in operation_list:
        iterations = operation_list[operation][1]
        for i in range(iterations):
            img = cv2_morphology_operation[operation](img, operation_list[operation])
    return img


def check_overlapping(input_point, start_point, end_point):
    return start_point[0] <= input_point[0] <= end_point[0] and start_point[1] <= input_point[1] <= end_point[1]


def mask_roi(img, dx=32, dy=74):
    img[0:dx, 0:dy] = 0
    return img


def processing_frame(frame):
    source_frame = frame.copy()
    frame = mask_roi(frame)
    # denoise_nn = predict_iter(frame, iterations=iterations)
    denoise_nn = np.clip(frame, 0, 1)
    # denoise_nlm = processing_cv((denoise_nn * 255 - 255).astype('uint8'))
    return denoise_nn, source_frame


def save_frame_as_tif(frame_index, frame, save_dir):
    plt.imsave(f'{save_dir}/frame_{frame_index}.jpg', frame, cmap='gray')
    return f'{save_dir}/frame_{frame_index}.jpg'


def get_distinct_objects(old_objects, current_objects):
    result_list = {}
    for (curr_object_ID, curr_centroid) in current_objects.items():
        if curr_object_ID not in old_objects.keys():
            result_list[curr_object_ID] = curr_centroid
    return result_list


def check_contour_overlapping(point, contour):
    return cv2.pointPolygonTest(contour, point, False)


def draw_ct_data(frame, res, frame_id, cont_list):
    for contour in cont_list[frame_id]:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
    if frame_id in res:
        for objects_data in res[frame_id]:
            obj_id = objects_data[0]
            centroid = objects_data[1]
            text = f"ID {obj_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    return frame


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


class VideoProcessing:

    def __init__(self):
        self.SAMPLE_NAME = Config.SAMPLE_NAME
        self.LOAD_VIDEO_PATH = f'{Config.SAMPLES_DIRECTORY}/{self.SAMPLE_NAME}.{Config.SAMPLE_EXTENSION}'
        self.FRAMES_SAVE_PATH = f'{Config.SAVE_FOLDER_PATH}/{self.SAMPLE_NAME}'
        self.SOURCE_FRAMES_SAVE_PATH = self.FRAMES_SAVE_PATH + '/source'
        self.DENOISED_FRAMES_SAVE_PATH = self.FRAMES_SAVE_PATH + '/nn_denoise'
        self.BINARY_FRAMES_SAVE_PATH = self.FRAMES_SAVE_PATH + '/morph'
        self.RESULT_FRAMES_SAVE_PATH = self.FRAMES_SAVE_PATH + '/contours_gray'
        self.VIDEO_SAVE_PATH = f"{Config.VIDEO_SAVE_FOLDER_PATH}/{self.SAMPLE_NAME}_compile.mp4"
        self.model = N2V(None, Config.N2V_MODEL_NAME, basedir=Config.N2V_MODEL_DIR)
        self.centroid_tracker = CentroidTracker()
        self.scale_factor = Config.SCALE_FACTOR

    def process(self):
        self.prepare_directories()
        ct_data, contours_list, ct_data_dict = self.get_data_denoised_video()
        mean_data = get_mean_data(ct_data, Config.MEAN_DATA_SAMPLE_RATE)
        inverse_dict = get_inverse_dict(mean_data)
        self.draw_contours_and_save(inverse_dict, contours_list)
        self.save_video_compilation()
        self.data_to_xlsx(ct_data_dict, f'{Config.XLSX_SAVE_PATH}/')

    def prepare_directories(self):
        save_path = f"./{Config.SAVE_FOLDER_PATH}/{self.SAMPLE_NAME}"
        Path(f"{save_path}/source").mkdir(parents=True, exist_ok=True)
        Path(f"{save_path}/nn_denoise").mkdir(parents=True, exist_ok=True)
        Path(f"{save_path}/morph").mkdir(parents=True, exist_ok=True)
        Path(f"{save_path}/contours_gray").mkdir(parents=True, exist_ok=True)
        Path(f"{Config.XLSX_SAVE_PATH}").mkdir(parents=True, exist_ok=True)
        Path(f"{Config.VIDEO_SAVE_FOLDER_PATH}").mkdir(parents=True, exist_ok=True)

    def binaryzation(self, img, ops_for_img, filters_params, debug_view=False, base_image=None):

        img_uint8 = np.uint8(img)
        nn_image = img_uint8.copy()

        if filters_params['d'] != 0:
            img_uint8 = cv2.bilateralFilter(
                img_uint8,
                filters_params['d'],
                filters_params['sigma_color'],
                filters_params['sigma_space'],
            )
        bilateral_img = img_uint8.copy()

        if filters_params['kernel_median'] != 0:
            img_uint8 = cv2.medianBlur(img_uint8, filters_params['kernel_median'])
        median_img = img_uint8.copy()

        if filters_params['kernel_gaussian'] != 0:
            img_uint8 = cv2.GaussianBlur(img_uint8, filters_params['kernel_gaussian'], 0)
        gaussian_img = img_uint8.copy()

        # img_binary = binary_segmentation(img_uint8.copy(), 95, 130, 255)
        img_binary = binary_segmentation(img_uint8.copy(), 200, 220, 255)
        binary_img = img_binary.copy()

        img_morph = cv_op(img_binary, ops_for_img)
        img_morph = set_clean_boundary(img_morph)
        morph_img = img_morph.copy()

        if debug_view:
            fig = plt.figure(figsize=(10, 7))
            rows = 2
            columns = 4

            fig.add_subplot(rows, columns, 1)
            plt.imshow(base_image, cmap="gray")
            plt.axis('off')
            plt.title("base_image")

            fig.add_subplot(rows, columns, 2)
            plt.imshow(nn_image, cmap="gray")
            plt.axis('off')
            plt.title("nn_image")

            fig.add_subplot(rows, columns, 3)
            plt.imshow(bilateral_img, cmap="gray")
            plt.axis('off')
            plt.title("bilateral_img")

            fig.add_subplot(rows, columns, 4)
            plt.imshow(median_img, cmap="gray")
            plt.axis('off')
            plt.title("median_img")

            fig.add_subplot(rows, columns, 5)
            plt.imshow(gaussian_img, cmap="gray")
            plt.axis('off')
            plt.title("gaussian_img")

            fig.add_subplot(rows, columns, 6)
            plt.imshow(binary_img, cmap="gray")
            plt.axis('off')
            plt.title("binary_img")

            fig.add_subplot(rows, columns, 7)
            plt.imshow(morph_img, cmap="gray")
            plt.axis('off')
            plt.title("morph_img")

            fig.add_subplot(rows, columns, 8)
            plt.imshow(morph_img, cmap="gray")
            plt.axis('off')
            plt.title("morph_img")
            plt.show()

        return img_morph, img_uint8

    def find_contours_only(self, img):
        img_uint8 = np.uint8(img)
        nn_predict = np.uint8(self.model.predict(img_uint8, axes="YX"))

        nlm = cv2.fastNlMeansDenoising(nn_predict,
                                       h=Config.NLM_H,
                                       templateWindowSize=Config.NLM_TEMPLATE_WINDOW_SIZE,
                                       searchWindowSize=Config.NLM_TEMPLATE_WINDOW_SIZE)

        img_bilateral = cv2.bilateralFilter(nlm,
                                            Config.BILATERAL_D,
                                            Config.BILATERAL_SIGMA_COLOR,
                                            Config.BILATERAL_SIGMA_SPACE)

        img_median = cv2.medianBlur(img_bilateral, Config.MEDIAN_K)

        if Config.USE_ADAPTIVE:
            (T, img_binary) = cv2.threshold(img_median, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            img_binary = binary_segmentation(img_median, Config.BINARY_THRESHOLD_MIN, Config.BINARY_THRESHOLD_MAX, 255)

        ops_for_img = Config.MORPH_OPERATIONS

        img_morph = cv_op(img_binary, ops_for_img)
        img_morph = set_clean_boundary(img_morph)
        img_morph = cv2.GaussianBlur(img_morph, Config.GAUSSIAN_K, 0)

        contours, hierarchy = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        approx_contours = []
        for cnt in contours:
            epsilon = Config.CONTOUR_EPSILON * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx_contours.append(approx)
        valid_contours = []
        new_hierarchy = []
        mask_holes = np.zeros(img_morph.shape[:2], np.uint8)
        for i, contour in enumerate(approx_contours):
            if len(contour) > Config.CONTOUR_MIN_LENGTH and cv2.contourArea(contour) > Config.CONTOUR_MIN_AREA:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area
                if hierarchy[0][i][3] != -1:
                    cv2.drawContours(mask_holes, [contour], -1, 255, -1)
                if hierarchy[0][i][3] != -1 or solidity <= Config.CONTOUR_MIN_SOLIDITY:
                    continue
                contour_moments = cv2.moments(contour)
                center_x = int(contour_moments["m10"] / contour_moments["m00"])
                center_y = int(contour_moments["m01"] / contour_moments["m00"])
                if not check_overlapping((center_x, center_y), (0, 0), (2 * 37, 2 * 16)):
                    valid_contours.append(contour)
                    new_hierarchy.append(hierarchy[0][i])
        # TRACKING
        bounding_rectangles = []
        for contour in valid_contours:
            mask = np.zeros(img_morph.shape[:2], np.uint8)
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
        return valid_contours, img_morph

    def get_data_denoised_video(self):
        cap = cv2.VideoCapture(self.LOAD_VIDEO_PATH)
        frame_counter = 0
        contours_data_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break

            # DataPreprocessing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:-36, :]
            frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

            # Denoise
            processed_frame, source_frame = processing_frame(frame.astype('float32') / 255)

            # Binarization
            contours_data, frame_binary = self.find_contours_only((processed_frame * 255).astype('uint8'))
            contours_data_list.append(contours_data)

            # SavingMidData
            save_frame_as_tif(frame_counter, source_frame, self.SOURCE_FRAMES_SAVE_PATH)
            save_frame_as_tif(frame_counter, processed_frame, self.DENOISED_FRAMES_SAVE_PATH)
            save_frame_as_tif(frame_counter, frame_binary, self.BINARY_FRAMES_SAVE_PATH)

            # Tracking
            prev_objects_id = self.centroid_tracker.objects.copy()
            data_dict = self.centroid_tracker.data_dict
            new_objects_id = self.centroid_tracker.objects
            new_objects = get_distinct_objects(prev_objects_id, new_objects_id)

            for item in new_objects.keys():
                for prev_item in prev_objects_id.keys():
                    prev_item_contour = data_dict[prev_item][len(data_dict[prev_item]) - 2]['centroid_data'][2]
                    res = check_contour_overlapping((
                        int(new_objects[item][0]),
                        int(new_objects[item][1])),
                        prev_item_contour,
                    )
                    if res == 1:
                        data_dict[item][0]['parent'].append(prev_item)
                        break
            frame_counter += 1

        cap.release()

        return self.centroid_tracker.data, contours_data_list, self.centroid_tracker.data_dict

    def draw_contours_and_save(self, result_dict, cont_list):
        cap = cv2.VideoCapture(self.LOAD_VIDEO_PATH)
        frame_counter = 0
        while cap.isOpened:
            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:-36, :]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            frame = draw_ct_data(frame, result_dict, frame_counter, cont_list)
            save_frame_as_tif(frame_counter, frame, self.RESULT_FRAMES_SAVE_PATH)
            frame_counter += 1

        cap.release()

    def save_video_compilation(self):
        fps = Config.OUT_VIDEO_FPS
        input_params = {'-r': fps, }
        output_params = {'-r': fps, '-vcodec': 'libx264', '-crf': '0'}
        writer = skvideo.io.FFmpegWriter(self.VIDEO_SAVE_PATH, inputdict=input_params, outputdict=output_params)
        cap = cv2.VideoCapture(self.LOAD_VIDEO_PATH)
        frame_counter = 0

        while cap.isOpened:
            ret, frame = cap.read()
            if frame is None:
                break
            frame_1 = frame[:-36, :]

            frame_2 = cv2.imread(f'{self.DENOISED_FRAMES_SAVE_PATH}/frame_{frame_counter}.jpg')
            frame_2 = cv2.resize(frame_2, (0, 0), fx=1 / self.scale_factor, fy=1 / self.scale_factor)

            frame_3 = cv2.imread(f'{self.BINARY_FRAMES_SAVE_PATH}/frame_{frame_counter}.jpg')
            frame_3 = cv2.resize(frame_3, (0, 0), fx=1 / self.scale_factor, fy=1 / self.scale_factor)

            frame_4 = cv2.imread(f'{self.RESULT_FRAMES_SAVE_PATH}/frame_{frame_counter}.jpg')
            frame_4 = cv2.resize(frame_4, (0, 0), fx=1 / self.scale_factor, fy=1 / self.scale_factor)

            temp_img_row_1 = np.concatenate((frame_1, frame_2), axis=1)
            temp_img_row_2 = np.concatenate((frame_3, frame_4), axis=1)
            result_image = np.concatenate((temp_img_row_1, temp_img_row_2), axis=0)

            writer.writeFrame(result_image)
            frame_counter += 1

        cap.release()
        writer.close()

    def data_to_xlsx(self, data_dict, output_path=''):
        frames_id, objects_id, center_x, center_y, centroid_s, centroid_p, solidity, spd_x, spd_y = [[] for x in
                                                                                                     range(9)]
        for (key, val) in data_dict.items():
            for data in val:
                objects_id.append(key)
                frames_id.append(data['frame_id'] + 1)
                center_x.append(data["center_xy"][0])
                center_y.append(data["center_xy"][1])
                centroid_s.append(data["centroid_data"][0])
                centroid_p.append(data["centroid_data"][1])
                solidity.append(data["centroid_data"][3])
                spd_x.append(data["speed_vector"][0])
                spd_y.append(data["speed_vector"][1])

        df = pd.DataFrame(
            {
                'ObjectID': objects_id,
                'Time(frame)': frames_id,
                'cX(px)': center_x,
                'cY(px)': center_y,
                'S(px^2)': centroid_s,
                'P(px)': centroid_p,
                'Solidity': solidity,
                'SpdX(px/frame)': spd_x,
                'SpdY(px/frame)': spd_y,
            }
        )
        df.to_excel(f"{output_path + self.SAMPLE_NAME}_result_1.xlsx", sheet_name="processing_result", index=False)
        df = df.groupby(['Time(frame)'])['ObjectID'].agg(['count'])
        df = df.rename(columns={'count': 'ObjectCount'})
        df.to_excel(f"{output_path + self.SAMPLE_NAME}_result_2.xlsx", sheet_name="processing_result", index=True)
