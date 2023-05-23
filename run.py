from video_processing import VideoProcessing

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

    'USE_NLM': True,
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

if __name__ == '__main__':
    vp = VideoProcessing(config)
    vp.process()
