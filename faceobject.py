# -*- coding: utf-8 -*-
import cv2
import numpy as np


class face_object:
    # 初始
    def __init__(self):
        self.face_no = [0]  # face id
        self.face_no_count = 0  # tracker no
        self.ix = [0]  # face rectangular position central x current
        self.iy = [0]  # face rectangular position central y current
        self.w = [0]  # face rectangular position central w current
        self.w_max = 100  # face rectangular position central h max current
        self.h = [0]  # face rectangular position central h current
        self.face_name = ['']  # face name show in each tracker
        self.face_name_list = [{'none': 0}]  # face name count in each tracker
        self.face_name_count_max = [0]  # face name count max in each tracker
        self.face_image_data = ['']  # face image data
        self.center_x = [0]  # face rectangular position central x latest
        self.center_y = [0]  # face rectangular position central y latest
        self.distance = 4  # face rectangular position distance, 1/distance value high , distance close
        self.face_count = 0  # face count
        self.face_no_time = [0]  # face position cumulative number
        self.face_no_time_max = 40  # face position cumulative number max
        self.face_no_run = [0]  # face still catch
        self.add_values = 2  # face cumulative number
        self.add_values_max = 60  # face cumulative number max

        self.trackers = {} # a dict to stash trackers. { key:index, value:tracker object}

    def tracker_init(self, init_value):
        self.face_no_count = init_value

    # 將每個 frame 的 box 輸入
    def detect_face(self, face_count, n_ix, n_iy, n_w, n_h, n_face_name, face_image_data):
        for i in range(0, face_count):
            self.detect_same_box(n_ix[i], n_iy[i], n_w[i], n_h[i], n_face_name[i], face_image_data[i])

    # 偵測框是否相近
    def detect_same_box(self, n_ix, n_iy, n_w, n_h, n_face_name, n_face_image_data):
        n_center_x = n_ix + n_w/2
        n_center_y = n_iy + n_h/2
        multi_index = []
        multi_center_x = []
        multi_center_y = []
        for i in range(1, self.face_count+1):
            if abs(n_center_x-self.center_x[i]) <= (self.w[i]/self.distance) and \
               abs(n_center_y-self.center_y[i]) <= (self.h[i]/self.distance):
                multi_index.append(i)
                multi_center_x.append(abs(n_center_x-self.center_x[i]))
                multi_center_y.append(abs(n_center_y-self.center_y[i]))
        if len(multi_index) > 0:
            if sorted(multi_center_x) < sorted(multi_center_y):
                out_index = sorted(range(len(multi_center_x)), key=lambda k: multi_center_x[k])
            else:
                out_index = sorted(range(len(multi_center_y)), key=lambda k: multi_center_x[k])
            face_count_index = multi_index[out_index[0]]
            # data output
            if self.face_no_time[face_count_index] < self.face_no_time_max:
                self.face_no_time[face_count_index] = self.face_no_time[face_count_index] + 1
            face_no_index = self.face_no[face_count_index]
            self.update_face(face_no_index, n_ix, n_iy, n_w, n_h, n_face_name, n_face_image_data)
            self.face_no_run[face_count_index] = 1
        else:
            self.add_face(n_ix, n_iy, n_w, n_h, n_face_name, n_face_image_data)

    # 新增框
    def add_face(self, ix, iy, w, h, face_name, face_image_data):
        self.face_no_count = self.face_no_count + 1
        self.face_no.append(self.face_no_count)
        self.face_count = self.face_count + 1
        self.ix.append(ix)
        self.iy.append(iy)
        self.w.append(w)
        self.h.append(h)
        self.face_name.append(face_name)
        self.face_name_count_max.append(0)
        self.face_name_list.append({face_name: self.add_values})
        self.face_image_data.append(face_image_data)
        self.center_x.append(ix + w/2)
        self.center_y.append(iy + h/2)
        self.face_no_time.append(1)
        self.face_no_run.append(1)

    # 刪除框
    def del_face(self, face_no):
        del_index = self.face_no.index(face_no)
        self.face_count = self.face_count - 1
        self.face_no.pop(del_index)
        self.ix.pop(del_index)
        self.iy.pop(del_index)
        self.w.pop(del_index)
        self.h.pop(del_index)
        self.face_name.pop(del_index)
        self.face_name_count_max.pop(del_index)
        self.face_name_list.pop(del_index)
        self.face_image_data.pop(del_index)
        self.center_x.pop(del_index)
        self.center_y.pop(del_index)
        self.face_no_time.pop(del_index)
        self.face_no_run.pop(del_index)

    # 更新框
    def update_face(self, face_no, ix, iy, w, h, face_name, face_image_data):
        update_index = self.face_no.index(face_no)
        self.ix[update_index] = ix
        self.iy[update_index] = iy
        self.w[update_index] = w
        self.h[update_index] = h
        # update face_name from face list
        self.update_face_name_list(update_index, face_name)
        self.face_image_data[update_index] = face_image_data
        self.center_x[update_index] = ix + w/2
        self.center_y[update_index] = iy + h/2

    # 檢查臉 name 是否持續一段時間.
    def update_face_name_list(self, update_index, face_name):
        face_name_list_select = self.face_name_list[update_index]

        input_face_name = face_name
        if input_face_name in face_name_list_select:
            if self.w[update_index] > self.w_max:
                face_name_list_select[input_face_name] = face_name_list_select[input_face_name] + self.add_values
            if face_name_list_select[input_face_name] > self.add_values_max:
                face_name_list_select[input_face_name] = self.add_values_max
        else:
            face_name_list_select[input_face_name] = self.add_values

        face_name_max_no = ['', 0]
        del_list = []
        for val in face_name_list_select:
            if face_name_list_select[val] > face_name_max_no[1]:
                face_name_max_no[0] = val
                face_name_max_no[1] = face_name_list_select[val]

            face_name_list_select[val] = face_name_list_select[val] - 1

            if face_name_list_select[val] <= 0:
                del_list.append(val)

        for val in del_list:
            del face_name_list_select[val]

        self.face_name[update_index] = face_name_max_no[0] #output
        self.face_name_count_max[update_index] = int(face_name_max_no[1])

    # 檢查臉是否持續一段時間
    def check_face(self, no_face=None):
        del_list = []
        for i in range(1, self.face_count + 1):
            if self.face_no_run[i] == 0:
                self.face_no_time[i] = self.face_no_time[i] - 1
            else:
                self.face_no_run[i] = 0

            if self.face_no_time[i] < 0:
                del_list.append(self.face_no[i])

        for i in del_list:
            self.del_face(i)
            if no_face:
                del (no_face[i])

    def draw_boxes(self, image, color=(0, 0, 255)):

        for i in range(1, self.face_count + 1):
            tracker_x = float(self.ix[i])
            tracker_y = float(self.iy[i])
            tracker_w = float(self.w[i])
            tracker_h = float(self.h[i])

            xmin = int((tracker_x - tracker_w / 2))
            xmax = int((tracker_x + tracker_w / 2))
            ymin = int((tracker_y - tracker_h / 2))
            ymax = int((tracker_y + tracker_h / 2))

            cv2.putText(image, str(self.face_no[i]) + ' (' + str(self.face_no_time[i]) + ')',
                        (xmin, ymax - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        (0, 255, 0), 2)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)

        return image


def get_img_by_tracker_box(box, small_img, real_size_img, re_scale):
    tracker_x = float(box[0])  # center point of tracker box
    tracker_y = float(box[1])
    tracker_w = float(box[2])
    tracker_h = float(box[3])

    box_wmin = np.max([int((tracker_x - tracker_w / 2)), 0])
    box_wmax = np.min([int((tracker_x + tracker_w / 2)), small_img.shape[1]])
    box_hmin = np.max([int((tracker_y - tracker_h / 2)), 0])
    box_hmax = np.min([int((tracker_y + tracker_h / 2)), small_img.shape[0]])

    # east+ocr
    box_of_real_size_img_wmin = box_wmin * re_scale
    box_of_real_size_img_wmax = box_wmax * re_scale
    box_of_real_size_img_hmin = box_hmin * re_scale
    box_of_real_size_img_hmax = box_hmax * re_scale
    sub_img = real_size_img[box_of_real_size_img_hmin:box_of_real_size_img_hmax,
              box_of_real_size_img_wmin:box_of_real_size_img_wmax, :]
    # show plate
    # cv2.imshow('track_no_%s' % track[4], sub_img)
    return sub_img
