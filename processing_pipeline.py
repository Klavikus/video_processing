import cv2


class ImageProcessingPipeline:
    def __init__(self):
        self.filters = []
        self.filter_data = {}

    def add_filter(self, image_filter):
        self.filters.append(image_filter)

    def process_sequence(self, sequence, debug_show=False):
        processed_sequence = []
        for frame in sequence:
            processed_frame = frame
            for image_filter in self.filters:
                processed_frame = image_filter.process(processed_frame)
                filter_type = type(image_filter).__name__
                if filter_type not in self.filter_data:
                    self.filter_data[filter_type] = []
                self.filter_data[filter_type].append(image_filter.filter_data)
            processed_sequence.append(processed_frame)
        return processed_sequence

    def process_sequence_from_video(self, video_path, debug=False, take=-1):
        processed_sequence = []
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
                    self.filter_data[filter_type].append(image_filter.filter_data)
                processed_sequence.append(processed_frame)

                if debug:
                    print(f'Frame proceed: {current_frame_id}')
                current_frame_id += 1
                if current_frame_id == take:
                    break
            else:
                break

        cap.release()

        return processed_sequence
