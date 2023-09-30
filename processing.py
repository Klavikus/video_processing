from data_processing import *
from image_filters import *
from processing_pipeline import ImageProcessingPipeline
from object_tracker import CentroidTracker
from unet_model import UnetModel

config = {
    'SAMPLE_NAME': 'Mov_S1',
    'SAMPLE_EXTENSION': 'avi',
    'SAMPLES_DIRECTORY': 'data_for_denoise',
    'SAVE_FOLDER_PATH': 'results',
    'VIDEO_SAVE_FOLDER_PATH': 'video_out',
    'XLSX_SAVE_PATH': 'xlsx_result',

    'MEAN_DATA_SAMPLE_RATE': 30,

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

SAMPLE_NAME = 'Mov_S1'
SAMPLE_EXTENSION = 'avi'
SAMPLES_DIRECTORY = 'data_for_denoise'

VIDEO_PATH = f'{SAMPLES_DIRECTORY}/{SAMPLE_NAME}.{SAMPLE_EXTENSION}'

model = UnetModel().compile('best_model.h5')

pipeline = ImageProcessingPipeline()
# pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/source', 'frame'))
pipeline.add_filter(CropFilter(config, x_slice, y_slice))  # Обрезаем легенду снизу
pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/sliced', f'{SAMPLE_NAME}_frame'))
pipeline.add_filter(ReplaceFilter(config, replace_x_slice, replace_y_slice))  # Закрашиваем чёрным надпись RUN
pipeline.add_filter(ConverterBGR2GRAY(config))  # Конвертируем в gray из bgr
pipeline.add_filter(DenoiseFilter(config))  # Шумоподавление
# pipeline.add_filter(SaveAsPng(config, f'results/{SAMPLE_NAME}/denoised', f'{SAMPLE_NAME}_frame'))
pipeline.add_filter(NeuroBinary(config, model))  # Бинаризация NN
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

import pandas as pd

def data_to_xlsx(data_dict, output_path=''):
    frames_id, objects_id, center_x, center_y, centroid_s, centroid_p, solidity, spd_x, spd_y = [[] for _ in
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
    df.to_excel(f"{output_path + SAMPLE_NAME}_result_1.xlsx", sheet_name="processing_result", index=False)
    df = df.groupby(['Time(frame)'])['ObjectID'].agg(['count'])
    df = df.rename(columns={'count': 'ObjectCount'})
    df.to_excel(f"{output_path + SAMPLE_NAME}_result_2.xlsx", sheet_name="processing_result", index=True)


data_to_xlsx(ct_data_dict, f"{config['XLSX_SAVE_PATH']}/")