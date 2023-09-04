from data_processing import *
from image_filters import *
from processing_pipeline import ImageProcessingPipeline
from object_tracker import CentroidTracker

config = {
    'SAMPLE_NAME': 'Mov_S8',
    'SAMPLE_EXTENSION': 'mov',
    'SAMPLES_DIRECTORY': 'data_for_denoise',
    'SAVE_FOLDER_PATH': 'results',
    'VIDEO_SAVE_FOLDER_PATH': 'video_out',
    'XLSX_SAVE_PATH': 'xlsx_result',

    'SCALE_FACTOR': 2,
    'MEAN_DATA_SAMPLE_RATE': 30,

    'BILATERAL_D': 3,
    'BILATERAL_SIGMA_COLOR': 25,
    'BILATERAL_SIGMA_SPACE': 25,

    'GAUSSIAN_K': 13,
    'MEDIAN_K': 7,

    'USE_ADAPTIVE': False,
    'BINARY_THRESHOLD_MIN': 56,
    'BINARY_THRESHOLD_MAX': 250,

    'USE_NLM': False,
    'NLM_H': 8,
    'NLM_TEMPLATE_WINDOW_SIZE': 37,
    'NLM_SEARCH_WINDOW_SIZE': 15,

    'MORPH_OPERATIONS': [
        # "close_3_1",
        # "erode_3_1",
        "open_5_1",
        # "dilate_3_1",
    ],

    'CONTOUR_EPSILON': 0.0001,
    'CONTOUR_MIN_LENGTH': 5,
    'CONTOUR_MIN_AREA': 150,
    'CONTOUR_MIN_SOLIDITY': 0.8,
    'OUT_VIDEO_FPS': 29.97,
}

x_slice = (0, -36)
y_slice = (0, 0)

replace_x_slice = (0, 34)
replace_y_slice = (0, 74)

SAMPLE_NAME = 'S1'
VIDEO_PATH = 'data_for_denoise/Mov_S1.avi'

pipeline = ImageProcessingPipeline()
# pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/source', 'frame'))
pipeline.add_filter(CropFilter(config, x_slice, y_slice))  # Обрезаем легенду снизу
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/sliced', f'{SAMPLE_NAME}_frame'))
pipeline.add_filter(ReplaceFilter(config, replace_x_slice, replace_y_slice))  # Закрашиваем чёрным надпись RUN
pipeline.add_filter(ConverterBGR2GRAY(config))  # Конвертируем в gray из bgr
pipeline.add_filter(DenoiseFilter(config))  # Шумоподавление
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/denoised', f'{SAMPLE_NAME}_frame'))
pipeline.add_filter(BinaryzationFilter(config))  # Бинаризация
pipeline.add_filter(MorphologicalFilter(config))
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/binaryzed', f'{SAMPLE_NAME}_frame'))
pipeline.add_filter(ContourDetectionFilter(config))  # Определение контуров
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/with_contours', f'{SAMPLE_NAME}_frame'))

processed_sequence = pipeline.process_sequence_from_video(VIDEO_PATH, debug=True, take=-1)

# print(processed_sequence['BinaryzationFilter'])
print(processed_sequence.keys())
filter_data = pipeline.filter_data

centroid_tracker = CentroidTracker()
tracking_module = TrackingModule(centroid_tracker)

frame_amount = len(processed_sequence['DenoiseFilter'])

contours_list = []

for i in range(frame_amount):
    image = processed_sequence['DenoiseFilter'][i]
    contours = filter_data['ContourDetectionFilter'][i]['contours']
    mask_holes = filter_data['ContourDetectionFilter'][i]['mask_holes']
    tracking_module.process(image, contours, mask_holes)
    contours_list.append(contours)

ct_data = centroid_tracker.get_data()
ct_data_dict = centroid_tracker.get_data_dict()

mean_data = get_mean_data(ct_data, config['MEAN_DATA_SAMPLE_RATE'])
inverse_dict = get_inverse_dict(mean_data)
draw_contours_and_save(inverse_dict, contours_list, VIDEO_PATH, f'results/{SAMPLE_NAME}/with_contours',
                       with_cv2_pause=False)

print(centroid_tracker.get_data())

