import matplotlib.pyplot as plt
import cv2


def calc_mean(seq, fps, start_index, length):
    temp_mean = [0, 0]
    for i in range(length):
        temp_elem = seq[start_index + i][3]
        temp_mean = [temp_mean[0] + temp_elem[0], temp_mean[1] + temp_elem[1]]
    return [temp_mean[0] / length * fps, temp_mean[1] / length * fps]


def replace_by_descriptor(seq_by_id, descriptor_list, fps):
    result_list = []
    for descriptor in descriptor_list:
        start_index = descriptor[1]
        len_seq = descriptor[0]
        if len_seq == 0:
            continue
        temp_mean = calc_mean(seq_by_id, fps, start_index, len_seq)
        for i in range(len_seq):
            old_data = seq_by_id[start_index + i]
            new_data = [old_data[0], old_data[1], old_data[2], temp_mean]
            result_list.append(new_data)
    return result_list


def find_consistent_seq_descriptor(input_list, fps):
    frame_counter = 0
    prev_frame_index = input_list[0][0] - 1
    descriptor = []
    for i, elem in enumerate(input_list):
        if prev_frame_index == elem[0] - 1:
            frame_counter += 1
        else:
            descriptor.append([frame_counter, i - frame_counter])
            frame_counter = 1
        if frame_counter == fps:
            descriptor.append([fps, i + 1 - fps])
            frame_counter = 0
        elif i == len(input_list) - 1:
            descriptor.append([frame_counter, i + 1 - frame_counter])
            frame_counter = 0
        prev_frame_index = elem[0]
    return descriptor


def get_mean_data(data, fps):
    result_dict = {}
    for key, val in data.items():
        if not val:
            continue
        descriptor = find_consistent_seq_descriptor(val, fps)
        temp_list = replace_by_descriptor(val, descriptor, fps)
        result_dict[key] = temp_list
    return result_dict


def get_inverse_dict(input_dict):
    result_dict = {}
    for key, val in input_dict.items():
        for frame_data in val:
            if frame_data[0] in result_dict:
                result_dict[frame_data[0]] += [(key, frame_data[1], frame_data[2], frame_data[3])]
            else:
                result_dict[frame_data[0]] = [(key, frame_data[1], frame_data[2], frame_data[3])]
    return result_dict


def draw_contours_and_save(result_dict, cont_list, load_video_path, result_frames_save_path):
    cap = cv2.VideoCapture(load_video_path)
    frame_counter = 0
    while cap.isOpened:
        ret, frame = cap.read()
        if frame is None:
            break

        frame = draw_ct_data(frame, result_dict, frame_counter, cont_list)
        save_frame_as_png(frame_counter, frame, result_frames_save_path)
        cv2.namedWindow("Input")
        cv2.imshow("Input", frame)
        cv2.waitKey(0)
        frame_counter += 1

    cap.release()


def save_frame_as_png(frame_index, frame, save_dir):
    plt.imsave(f'{save_dir}/frame_{frame_index}.png', frame, cmap='gray')
    return f'{save_dir}/frame_{frame_index}.png'


def draw_ct_data(frame, res, frame_id, cont_list):
    if frame_id >= len(cont_list):
        return frame
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
