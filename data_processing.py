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
