import numpy as np
import os
import cv2
import requests
from config.mongodb import Mongodb
from config.configure import Config

from PIL import Image, ImageDraw, ImageFont

configure = Config(os.path.abspath(os.path.join(os.path.curdir)))
nodejs_CarStatistics_url = configure.nodejs_CarStatistics_url
CarStatistics_dict = {}

mongodb = Mongodb()
mongodblen = mongodb.search_data()
max_seq_no = int(mongodblen[0]['flow_no']) if mongodblen else 0
# max_seq_no = 0


def draw_boxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), color, 3)
    return image


def puttext_in_chinese(img, text, location, color=(0, 0, 255), fontsize=40):
    # cv2 to pil
    cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)

    # text
    draw = ImageDraw.Draw(pil_img)
    # font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    # draw.text(location, text, color, font=font)  # third parameter is color
    draw.text(location, text, color)  # third parameter is color

    # pil to cv2
    cv2_text_im = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_text_im


def draw_boxes_by_center(image, boxes, seq_no, color=(255, 50, 50), font_size=6):
    tracker_x = float(boxes[0])
    tracker_y = float(boxes[1])
    tracker_w = float(boxes[2])
    tracker_h = float(boxes[3])

    xmin = int((tracker_x - tracker_w / 2))
    xmax = int((tracker_x + tracker_w / 2))
    ymin = int((tracker_y - tracker_h / 2))
    ymax = int((tracker_y + tracker_h / 2))

    cv2.putText(image, str(seq_no),
                (xmin, ymax - 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size * 1e-3 * image.shape[0],
                (0, 255, 0), thickness=8)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
    return image


def submit(url, json):
    headers = {'Content-type': 'application/json'}
    resp = requests.post(url, json=json, headers=headers)
    return resp


def stop_car_recognition(frame, display_frame, tracker, stop_car, car_flow, point_polylines, stop_flag, track_no_path,
                         threashold_of_stopping=30):
    global nodejs_CarStatistics_url
    global CarStatistics_dict
    track = tracker
    tracker_x = float(track[0])
    tracker_y = float(track[1])
    tracker_w = float(track[2])
    tracker_h = float(track[3])

    xmin = int((tracker_x - tracker_w / 2))
    xmax = int((tracker_x + tracker_w / 2))
    ymin = int((tracker_y - tracker_h / 2))
    ymax = int((tracker_y + tracker_h / 2))
    xstop = (xmin + xmax) / 2
    ystop = ymax

    list_xpoint = [point_polylines[0][0][0], point_polylines[1][0][0], point_polylines[2][0][0],
                   point_polylines[3][0][0]]
    list_ypoint = [point_polylines[0][0][1], point_polylines[1][0][1], point_polylines[2][0][1],
                   point_polylines[3][0][1]]
    if xstop > min(list_xpoint) and xstop < max(list_xpoint) and ystop > min(list_ypoint) and ystop < max(list_ypoint):
        # give a sequence number as its car_no
        if track[4] not in car_flow:
            global max_seq_no
            car_flow[track[4]] = [max_seq_no + 1, 0]
            # car_flow[track_no][0]: index start from 1, not 0
            # car_flow[track_no][1]: 0(illegal) or 1(legal)

            max_seq_no += 1

        # saved_flag = False
        if track[4] in stop_car.keys():
            stop_car[track[4]] += 1
        else:
            stop_car[track[4]] = 1

        if stop_car[track[4]] >= threashold_of_stopping:
            car_flow[track[4]][1] = 1
            CarStatistics_dict[car_flow[track[4]][0]] = car_flow[track[4]][1]
            stop_flag = True
        else:
            stop_flag = False
            # save_frame = puttext_in_chinese(frame, '未停車再開', (xstop, ymax - 30), (255, 0, 255), 20)
            # save_frame = draw_boxes_by_center(frame, tracker, car_flow[track[4]][0], color=(50, 50, 255))

            # cv2.imwrite(os.path.join(track_no_path, "%s.png" % (track[4])), save_frame)
        if len(CarStatistics_dict) <= 7:
            if car_flow[track[4]][0] not in CarStatistics_dict.keys():
                CarStatistics_dict[car_flow[track[4]][0]] = car_flow[track[4]][1]
        else:
            CarStatistics_dict.pop(np.min(list(CarStatistics_dict.keys())))
            CarStatistics_dict[car_flow[track[4]][0]] = car_flow[track[4]][1]

        submit(nodejs_CarStatistics_url, {"car_no": CarStatistics_dict})

    if track[4] in car_flow:
        display_frame = puttext_in_chinese(display_frame, str(car_flow[track[4]][0]), (xstop, ymax), (255, 0, 255),
                                           fontsize=60)

    return display_frame, stop_flag, car_flow


def offline_stop_car_recognition(tracker, point_polylines, track_no_dir, frame=None, display_frame=None,
                         threashold_of_stopping=30):
    for track_idx in range(1, tracker.face_count + 1):
        tracker_x = float(tracker.ix[track_idx])
        tracker_y = float(tracker.iy[track_idx])
        tracker_w = float(tracker.w[track_idx])
        tracker_h = float(tracker.h[track_idx])

        xmin = int((tracker_x - tracker_w / 2))
        xmax = int((tracker_x + tracker_w / 2))
        ymin = int((tracker_y - tracker_h / 2))
        ymax = int((tracker_y + tracker_h / 2))
        xstop = (xmin + xmax) / 2
        ystop = ymax

        list_xpoint = [point_polylines[0][0][0], point_polylines[1][0][0], point_polylines[2][0][0],
                       point_polylines[3][0][0]]
        list_ypoint = [point_polylines[0][0][1], point_polylines[1][0][1], point_polylines[2][0][1],
                       point_polylines[3][0][1]]

        if xstop > min(list_xpoint) and xstop < max(list_xpoint) and \
                ystop > min(list_ypoint) and ystop < max(list_ypoint):
            # car_flow[track_no][0]: index start from 1, not 0
            # car_flow[track_no][1]: 0(illegal) or 1(legal)

            if tracker.face_image_data[track_idx]["flow_no"] == '':
                tracker.face_image_data[track_idx]["flow_no"] = max_seq_no + 1

            tracker.face_image_data[track_idx]["stop_sec_counter"] += 1

            if tracker.face_image_data[track_idx]["stop_sec_counter"] >= threashold_of_stopping:
                tracker.face_image_data[track_idx]["stop_flag"] = True

                # CarStatistics_dict[car_flow[track[4]][0]] = car_flow[track[4]][1]
            # else:
            #     # save_frame = puttext_in_chinese(frame, '未停車再開', (xstop, ymax - 30), (255, 0, 255), 20)
            #     # save_frame = draw_boxes_by_center(frame, [tracker_x, tracker_y, tracker_w, tracker_h], tracker.face_image_data[track_idx]["flow_no"], color=(50, 50, 255))
            #     save_frame = frame
            #
            #     # save image of illegal car behavior
            #     if tracker.face_image_data[track_idx]["vanish_flag"]:
            #         track_no_path = os.path.join(track_no_dir, "track_no_%s" % str(track_idx))
            #         if not os.path.exists(track_no_path):
            #             os.makedirs(track_no_path)
            #         cv2.imwrite(os.path.join(track_no_path, "%s.png" % str(track_idx)), save_frame)
            # if len(CarStatistics_dict) <= 7:
            #     if car_flow[track[4]][0] not in CarStatistics_dict.keys():
            #         CarStatistics_dict[car_flow[track[4]][0]] = car_flow[track[4]][1]
            # else:
            #     CarStatistics_dict.pop(np.min(list(CarStatistics_dict.keys())))
            #     CarStatistics_dict[car_flow[track[4]][0]] = car_flow[track[4]][1]
            #
            # submit(nodejs_CarStatistics_url, {"car_no": CarStatistics_dict})

        elif tracker.face_image_data[track_idx]["stop_sec_counter"] >= 1:
            tracker.face_image_data[track_idx]["stop_sec_counter"] = 0
            save_video_path = os.path.join(track_no_dir,
                                               "track_no_%s" % str(track_idx))
            if not os.path.exists(save_video_path):
                os.makedirs(save_video_path)
            save_video_path = os.path.join(save_video_path,
                                           "%s.jpg" % str(track_idx))
            cv2.imwrite(save_video_path, display_frame)

        # if tracker.face_image_data[track_idx]["flow_no"] != '' and not isinstance(display_frame, type(None)):
        #     display_frame = puttext_in_chinese(display_frame,
        #                                        str(tracker.face_image_data[track_idx]["stop_sec_counter"]),
        #                                        (xstop, ymax),
        #                                        (255, 0, 255),
        #                                        fontsize=60)

    return display_frame
