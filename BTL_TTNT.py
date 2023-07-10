import os
import sys
import cv2
import math
from darknet import darknet
import numpy as np
from keras.models import load_model

video_path = sys.argv[1]
save_path = "/media/son/146D1F6B754FF5D0/Violent/test_newcam_new/no/"

class DetectParameter(object):
    def __init__(self, text, point1, point2, PrincipalPoint):
        self.text = text
        self.point1 = point1
        self.point2 = point2
        self.PrincipalPoint = PrincipalPoint

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

class ObjectDetect_outdoor(object):
    def __init__(self):
        configPath = "./model_yolo/yolov4.cfg"
        weightPath = "./model_yolo/yolov4.weights"
        metaPath = "./model_yolo/yolov4.data"
        self.netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
        self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain), darknet.network_height(self.netMain), 3)

    def detect_object(self, image):
        (Height_image, Width_image) = image.shape[:2]
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet.network_width(self.netMain), darknet.network_height(self.netMain)), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.5)
        res = []
        for detection in detections:
            x, y, w, h = int(detection[2][0]*Width_image / darknet.network_width(self.netMain)), int(detection[2][1]*Height_image/darknet.network_height(self.netMain)), int(detection[2][2]*Width_image / darknet.network_width(self.netMain)), int(detection[2][3]*Height_image/darknet.network_height(self.netMain))
            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
            text = detection[0].decode()
            if text == "person":
                obj = DetectParameter(text, (xmin, ymin), (xmax, ymax), (xmin + int((xmax-xmin)/2), ymin + int((ymax-ymin)/2)))
                res.append(obj)
        return res

def calculatedistane(brid_point1, brid_point2):
    dx2 = (brid_point1[0] - brid_point2[0])**2
    dy2 = (brid_point1[1] - brid_point2[1])**2
    distance = math.sqrt(dx2 + dy2)
    return distance

def calcualte_boxarea(tl_point, br_point):
    area = abs(tl_point[0] - br_point[0])*abs(tl_point[1] - br_point[1])
    return area

def merge_box(Box1, Box2):
    x_min = min(Box1.point1[0], Box2.point1[0])
    y_min = min(Box1.point1[1], Box2.point1[1])
    x_max = max(Box1.point2[0], Box2.point2[0])
    y_max = max(Box1.point2[1], Box2.point2[1])
    box_combine = DetectParameter("person", (x_min, y_min), (x_max, y_max), (x_min + int((x_max-x_min)/2), y_min + int((y_max-y_min)/2)))
    return box_combine

def calcualte_ratiobox(ratio_width, ratio_height, area_box, area_ratio, scale):
    height = int(abs(math.sqrt((area_ratio*area_box*ratio_height)/ratio_width)))
    width = int(height*ratio_width/ratio_height)
    return scale*height, scale*width

def classify_video(video_path):
    model = load_model("path/to/pretrained_model.h5")
    labels = ["non-violent", "violent"]

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    frames = np.array(frames)
    frames = frames / 255.0

    predictions = model.predict(frames)
    average_prediction = np.mean(predictions)

    if average_prediction >= 0.5:
        return labels[1]  # "violent"
    else:
        return labels[0]  # "non-violent"

def classify_single_person_nonviolent(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 1:
        return "non-violent"

    return classify_video(video_path)

def main():
    detect_Outdoor = ObjectDetect_outdoor()
    file_count = 0
    correct_detections = 0
    total_detections = 0
    single_person_nonviolent_count = 0

    for filename in os.listdir(video_path):
        if filename.endswith(".avi") or filename.endswith(".mp4"):
            file_count += 1
            print(os.path.join(video_path, filename))
            path = os.path.join(video_path, filename)

            classification = classify_single_person_nonviolent(path)

            if classification == "non-violent":
                single_person_nonviolent_count += 1

            cap = cv2.VideoCapture(path)
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

            cnt = 0
            count = 0
            frames = []
            res_D = []
            box_coordinates = {}
            frames_crop = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_detect = frame.copy()
                imageshow = frame.copy()
                image_crop = frame.copy()

                if cnt % 16 == 0:
                    count = count + 1
                    print(count)
                    res_D = detect_Outdoor.detect_object(image_detect)
                    box_coordinates = {}
                    frames_crop = {}

                if cnt % 16 == 15:
                    draw_coordiante= {}

                if len(res_D) == 0:
                    continue

                if len(res_D) != 0 and cnt % 16 == 0:
                    ClosedBox_group = {}
                    for k in range(len(res_D)):
                        alone = True
                        for h in range(k, len(res_D), 1):
                            if k != h:
                                new_group = False
                                distance_box = calculatedistane(res_D[k].PrincipalPoint, res_D[h].PrincipalPoint)
                                if distance_box < 120:
                                    if len(ClosedBox_group.keys()) == 0:
                                        ClosedBox_group[k] = [k, h]
                                    else:
                                        if k in ClosedBox_group.keys():
                                            ClosedBox_group[k].append(h)
                                        else:
                                            for group_index, group_box in ClosedBox_group.items():
                                                if k in group_box and h not in group_box:
                                                    ClosedBox_group[group_index].append(h)
                                                if h in group_box and k not in group_box:
                                                    ClosedBox_group[group_index].append(k)
                                            for group_index, group_box in ClosedBox_group.items():
                                                if k not in group_box and h not in group_box:
                                                    new_group = True
                                                else:
                                                    new_group = False
                                                    break
                                            if new_group:
                                                ClosedBox_group[k] = [k]

                    for index_box, group_box in ClosedBox_group.items():
                        box_combine = res_D[index_box]
                        area_person = calcualte_boxarea(res_D[index_box].point1, res_D[index_box].point2)
                        for i in range(len(group_box) - 1):
                            box_combine = merge_box(box_combine, res_D[group_box[i+1]])
                        height, width = calcualte_ratiobox(4, 3, area_person, 5, 1)
                        xmin = int(box_combine.point1[0] - (width - abs(box_combine.point2[0]-box_combine.point1[0]))/2)
                        ymin = int(box_combine.point1[1] - (height - abs(box_combine.point2[1]-box_combine.point1[1]))/2)
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        xmax = int(xmin + width)
                        ymax = int(ymin + height)
                        if xmax > w_frame:
                            xmax = int(w_frame)
                            xmin = int(xmax - width)
                        if ymax > h_frame:
                            ymax = int(h_frame)
                            ymin = int(ymax - height)
                        xcenter = int(box_combine.point1[0] + int((box_combine.point2[0]-box_combine.point1[0])/2))
                        ycenter = int(box_combine.point1[1] + int((box_combine.point2[1]-box_combine.point1[1])/2))
                        box_coordinates[index_box] = [xmin, ymin, xmax, ymax]
                        frames_crop[index_box] = []

                count_box = 0
                for index_box, coordinate_box in box_coordinates.items():
                    frame_crop = image_crop[coordinate_box[1]:coordinate_box[3], coordinate_box[0]:coordinate_box[2], :]
                    frame_crop = cv2.resize(frame_crop, (320, 240), interpolation=cv2.INTER_AREA)
                    frames_crop[index_box].append(frame_crop)

                    if len(frames_crop[index_box]) == 16:
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        out = cv2.VideoWriter(save_path + str(file_count) + "_" + str(count) + '_' + str(index_box) + '.avi', fourcc, fps, (320, 240))
                        for frame_tosave in frames_crop[index_box]:
                            out.write(frame_tosave)

                cnt = cnt + 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Evaluation
                total_detections += len(res_D)
                for detection in res_D:
                    if detection.text == "person":
                        correct_detections += 1

    accuracy = correct_detections / total_detections
    single_person_nonviolent_ratio = single_person_nonviolent_count / file_count

    print("Accuracy:", accuracy)
    print("Single Person Non-Violent Ratio:", single_person_nonviolent_ratio)

if __name__ == '__main__':
    main()
