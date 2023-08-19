from image_filters import *
from processing_pipeline import ImageProcessingPipeline
import matplotlib.image as mpimg


config = {
    'SAMPLE_NAME': 'Mov_S1',
    'SAMPLE_EXTENSION': 'avi',
    'SAMPLES_DIRECTORY': 'data_for_denoise',
    'SAVE_FOLDER_PATH': 'results',
    'VIDEO_SAVE_FOLDER_PATH': 'video_out',
    'XLSX_SAVE_PATH': 'xlsx_result',

    'SCALE_FACTOR': 2,
    'MEAN_DATA_SAMPLE_RATE': 30,

    'BILATERAL_D': 5,
    'BILATERAL_SIGMA_COLOR': 25,
    'BILATERAL_SIGMA_SPACE': 25,

    'GAUSSIAN_K': 7,
    'MEDIAN_K': 21,

    'USE_ADAPTIVE': True,
    'BINARY_THRESHOLD_MIN': 122,
    'BINARY_THRESHOLD_MAX': 150,

    'USE_NLM': False,
    'NLM_H': 8,
    'NLM_TEMPLATE_WINDOW_SIZE': 37,
    'NLM_SEARCH_WINDOW_SIZE': 15,

    'MORPH_OPERATIONS': [
        "close_3_1",
        "open_7_1",
        "dilate_3_1",
    ],

    'CONTOUR_EPSILON': 0.0001,
    'CONTOUR_MIN_LENGTH': 5,
    'CONTOUR_MIN_AREA': 380,
    'CONTOUR_MIN_SOLIDITY': 0.12,
    'OUT_VIDEO_FPS': 29.97,
}

x_slice = (0, -36)
y_slice = (0, 0)

replace_x_slice = (0, 34)
replace_y_slice = (0, 74)

SAMPLE_NAME = 'S1'
FILE_PATH = 'data_for_denoise/frames/frame_0.jpg'
VIDEO_PATH = 'data_for_denoise/Mov_S1.avi'

pipeline = ImageProcessingPipeline()
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/source', 'frame'))
pipeline.add_filter(CropFilter(config, x_slice, y_slice))  # Обрезаем легенду снизу
pipeline.add_filter(ReplaceFilter(config, x_slice, y_slice))  # Закрашиваем чёрным надпись RUN
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/sliced', 'frame'))
pipeline.add_filter(ConverterBGR2GRAY(config))  # Конвертируем в gray из bgr
pipeline.add_filter(DenoiseFilter(config))  # Шумоподавление
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/denoised', 'frame'))
pipeline.add_filter(BinaryzationFilter(config))  # Бинаризация
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/binaryzed', 'frame'))
pipeline.add_filter(ContourDetectionFilter(config))  # Определение контуров
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/with_contours', 'frame'))
pipeline.add_filter(ShowImage(config, 'gray'))

sequence = [mpimg.imread(FILE_PATH)]
# processed_sequence = pipeline.process_sequence(sequence, False)
processed_sequence = pipeline.process_sequence_from_video(VIDEO_PATH, debug=True, take=3)

# print(processed_sequence['BinaryzationFilter'])
print(processed_sequence.keys())
filter_data = pipeline.filter_data
# print(filter_data)
