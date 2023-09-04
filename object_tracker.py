from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np
import cv2


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30):
        """
        Инициализирует CentroidTracker.

        :param max_disappeared: Максимальное количество итераций, после которых объект считается потерянным.
        """
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.objects_vectors_instant = OrderedDict()
        self.disappeared = OrderedDict()
        self.tracked = OrderedDict()
        self.maxDisappeared = max_disappeared
        self.cur_frame = -1
        self.data = OrderedDict()
        self.data_dict = {}
        self.input_data = OrderedDict()
        self.parents = OrderedDict()

    def register(self, centroid: object, input_data):
        """
        Регистрирует новый объект для отслеживания.

        :param centroid: Центроид объекта (координаты X и Y).
        :param input_data: Входные данные для объекта.
        """
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.parents[self.nextObjectID] = ['none']
        self.data[self.nextObjectID] = []
        self.data_dict[self.nextObjectID] = [
            {
                'frame_id': self.cur_frame,
                'center_xy': centroid,
                'centroid_data': input_data,
                'speed_vector': (0, 0),
                'parent': self.parents[self.nextObjectID],
            }]
        self.tracked[self.nextObjectID] = 1
        self.objects_vectors_instant[self.nextObjectID] = (0, 0)
        self.nextObjectID += 1

    def unregister(self, object_id):
        """
        Удаляет объект из отслеживания.

        :param object_id: Уникальный идентификатор объекта.
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, bounding_rectangles):
        """
        Обновляет отслеживание на основе новых позиций объектов.

        :param bounding_rectangles: Список ограничивающих прямоугольников для объектов.
        :return: Обновленные координаты объектов и векторы скорости объектов.
        """
        if len(bounding_rectangles) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.maxDisappeared:
                    self.unregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(bounding_rectangles), 2), dtype="int")
        input_centroids_data = []

        for (i, (startX, startY, endX, endY, S, P, contour)) in enumerate(bounding_rectangles):
            center_x = int((startX + endX) / 2.0)
            center_y = int((startY + endY) / 2.0)
            input_centroids[i] = (center_x, center_y)
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            input_centroids_data.append((S, P, contour, solidity))

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], input_centroids_data[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            d = dist.cdist(np.array(object_centroids), input_centroids)
            rows = d.min(axis=1).argsort()
            cols = d.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if d[row][col] > 30:
                    continue

                object_id = object_ids[row]
                new_state = input_centroids[col]
                prev_state = self.objects[object_id]
                frame_delay = self.disappeared[object_id]
                vector_speed = self._calculate_speed(new_state, prev_state, frame_delay)

                self.objects_vectors_instant[object_id] = vector_speed
                self.tracked[object_id] += 1
                self.objects[object_id] = new_state
                self.disappeared[object_id] = 0
                self.data[object_id] += [(self.cur_frame, self.objects[object_id], input_centroids_data[col],
                                          self.objects_vectors_instant[object_id])]
                self.data_dict[object_id] += [
                    {
                        'frame_id': self.cur_frame,
                        'center_xy': self.objects[object_id],
                        'centroid_data': input_centroids_data[col],
                        'speed_vector': self.objects_vectors_instant[object_id],
                        'parent': self.parents[object_id],
                    }]
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, d.shape[0])).difference(used_rows)
            unused_cols = set(range(0, d.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.maxDisappeared:
                    self.unregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], input_centroids_data[col])

        self.cur_frame = self.cur_frame + 1
        return self.objects, self.objects_vectors_instant

    def get_data(self) -> OrderedDict:
        """
        Возвращает копию объекта `data`.
        """
        return self.data.copy()

    def get_data_dict(self) -> dict:
        """
        Возвращает копию объекта `data_dict`.
        """
        return self.data_dict.copy()

    @staticmethod
    def _calculate_speed(cur_state, prev_state, frame_delay):
        """
        Вычисляет вектор скорости на основе текущего и предыдущего состояний объекта.

        :param cur_state: Текущее состояние объекта (координаты X и Y).
        :param prev_state: Предыдущее состояние объекта (координаты X и Y).
        :param frame_delay: Задержка между кадрами.
        :return: Вектор скорости объекта.
        """
        vector_spd = (
            (cur_state[0] - prev_state[0]) / (1 + frame_delay), (cur_state[1] - prev_state[1]) / (1 + frame_delay))
        return vector_spd
