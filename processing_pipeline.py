import cv2


class ImageProcessingPipeline:
    def __init__(self):
        self.filters = []
        self.processed_sequence = {}
        self.filter_data = {}

    def add_filter(self, image_filter):
        self.filters.append(image_filter)

    def process_sequence_from_video(self, video_path, debug=False, take=-1):
        current_frame_id = 0
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error opening video stream or file")

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                processed_frame = frame
                for image_filter in self.filters:
                    processed_frame = image_filter.process(processed_frame)
                    filter_type = type(image_filter).__name__
                    if filter_type not in self.filter_data:
                        self.filter_data[filter_type] = []
                        self.processed_sequence[filter_type] = []
                    self.filter_data[filter_type].append(image_filter.filter_data)
                    self.processed_sequence[filter_type].append(processed_frame)
                if debug:
                    print(f'Frame proceed: {current_frame_id}')
                current_frame_id += 1
                if current_frame_id == take:
                    break
            else:
                break

        cap.release()

        return self.processed_sequence
