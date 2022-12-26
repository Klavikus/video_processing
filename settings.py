class Config:

    # Paths
    XLSX_SAVE_PATH = "xlsx_result"
    VIDEO_SAVE_FOLDER_PATH = "video_out"
    SAVE_FOLDER_PATH = "results"
    SAMPLES_DIRECTORY = "data_for_denoise"
    N2V_MODEL_NAME = "n2v_64x64_05_256_5"
    N2V_MODEL_DIR = "models"
    SAMPLE_NAME = "Mov_S5"
    SAMPLE_EXTENSION = "avi"

    # Params
    SCALE_FACTOR = 2
    MEAN_DATA_SAMPLE_RATE = 30

    # Filters params
    # NLM
    NLM_H = 3
    NLM_TEMPLATE_WINDOW_SIZE = 37
    NLM_SEARCH_WINDOW_SIZE = 15
    # Bilateral
    BILATERAL_D = 11
    BILATERAL_SIGMA_COLOR = 35
    BILATERAL_SIGMA_SPACE = 35
    # Median
    MEDIAN_K = 5
    # Gaussian
    GAUSSIAN_K = (13, 13)

    # Binarization
    USE_ADAPTIVE = False
    BINARY_THRESHOLD_MIN = 122
    BINARY_THRESHOLD_MAX = 150

    # Morphs operations
    MORPH_OPERATIONS = {
        'erode': ((3, 3), 1),
        'open': ((3, 3), 1),
        # 'close': ((7, 7), 1),
    }

    # Contour filter params
    CONTOUR_EPSILON = 0.0001
    CONTOUR_MIN_LENGTH = 5
    CONTOUR_MIN_AREA = 380
    CONTOUR_MIN_SOLIDITY = 0.12

    # Output video params
    OUT_VIDEO_FPS = "29.97"
