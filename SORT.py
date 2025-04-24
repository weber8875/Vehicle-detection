import cv2
import onnxruntime as ort
import numpy as np

from new_yolo_kalman_tracker import linear_sum_assignment


class ObjectTracker :
    def __init__(self, model_path, class_name_path, video_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        with open(class_name_path,"r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        self.trackers = []
        self.kalman_list = []
        self.id_list = []
        self.missed_frames = []
        self.next_id = 0

        self.MAX_MISSED = 10
        self.DETECT_EVERY = 5
        self.frame_id = 0

        self.crossed_ids = set()
        self.car_count = 0
        self.line_position = 30
        self.prev_centers = {}

    def create_kalman(self, x, y):
        kf = cv2.KalmanFilter(4,2)
        kf.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]], np.float32)
        kf.measurementMatrix = np.array([[1,0,0,0],
                                         [0,1,0,0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype = np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype = np.float32) * 1
        kf.statePre = np.array([[x],[y],[0],[0]], np.float32)
        return kf

    def iou(self, b1, b2):
        x1 = max(b1[0],b2[0])
        y1 = max(b1[1],b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = b1[2] * b1[3]
        area2 = b2[2] * b2[3]
        union = area1 + area2 - inter
        return inter/union if union > 0 else 0

    def detect_objects(self, frame):

        self.frame_id += 1
        h0, w0 = frame.shape[:2]
        detected = []

        if self.frame_id % self.DETECT_EVERY == 0:
            img_resized = cv2.resize(frame, (640,640))
            img_rgb = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
            img_input = img_rgb.astype(np.float32) / 255.0
            img_input = np.transpose(img_input, (2,0,1))[np.newaxis, :]

            outputs = self.session.run([self.output_name],{ self.input_name : img_input})[0]
            outputs = np.squeeze(outputs)

            for det in outputs:
                score = det[4]
                if score < 0.4: continue
                scores = det[5:]
                class_id = np.argmax(scores)
                label = self.class_names[class_id]
                confidence  = scores[class_id]

                if confidence < 0.5 or label not in ["car", "motorcycle", "bicycle"]: continue

                cx, cy, w, h = det[:4]
                x = int((cx - w / 2) * w0 / 640)
                y = int((cy - h / 2) * h0 / 640)
                w = int(w * w0 / 640)
                h = int(h * h0 / 640)
                detected.append([x,y,w,h,label])

        return detected

    def update_trackers(self, detected):
        predicted_boxes = []
        for kf in self.kalman_list:
            pred = kf.predict()
            px = int(pred[0].item())
            py = int(pred[1].item())
            predicted_boxes.append((px-20, py-20, 40, 40))

        cost_matrix = []
        for pb in predicted_boxes:
            row = []
            for db in detected:
                row.append(1 - self.iou(pb,db[:4]))
            cost_matrix.append(row)

        matched_indices = []
        unmatched_detections = set(range(len(detected)))
        unmatched_trackers = set(range(len(predicted_boxes)))

        if len(cost_matrix) > 0:
            cost_matrix = np.array(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r][c] < 0.3:
                    matched_indices.append((r, c))
                    unmatched_trackers.discard(r)
                    unmatched_detections.discard(c)

        for tracker_idx, det_idx  in matched_indices:
            x, y, w, h, _ = detected[det_idx]
            cx = x + w // 2
            cy = y + h // 2
            measurement = np.array([[np.float32(cx)],[np.float32(cy)]])
            self.kalman_list[tracker_idx].correct(measurement)
            self.trackers[tracker_idx] = detected[det_idx]
            self.missed_frames[tracker_idx] = 0

        for i in unmatched_trackers:
            self.missed_frames[i] += 1


        for i , (x, y, w, h, _) in enumerate(self.trackers):
            cx = x + w // 2
            cy = y + h // 2
            tid = self.id_list[i]
            prev = self.prev_centers.get(tid, None)
            if prev:
                if prev[1] < self.line_position and cy >= self.line_position:
                    if tid not in self.crossed_ids:
                        self.car_count += 1
                        self.crossed_ids.add(tid)
            self.prev_centers[tid] = (cx, cy)

        to_delete = [i for i, m in enumerate(self.missed_frames) if m > self.MAX_MISSED]
        for idx  in sorted(to_delete, reverse = True):
            del self. kalman_list[idx]
            del self.trackers[idx]
            del self.id_list[idx]
            del self.missed_frames[idx]


        for idx  in unmatched_detections:
            x, y, w, h, _ = detected[idx]
            cx = x + w // 2
            cy = y + h // 2
            kf = self.create_kalman(cx, cy)
            self.kalman_list.append(kf)
            self.trackers.append(detected[idx])
            self.id_list.append(self.next_id)
            self.missed_frames.append(0)
            self.next_id += 1

    def draw(self, frame):
        for i, bbox in enumerate(self.trackers):
            x, y, w, h, label = bbox
            pred = self.kalman_list[i].predict()
            px = int(pred[0].item())
            py = int(pred[1].item())

            color = (0, 255, 0) if label == "car" else (255, 0, 0)  # 不同類別不同顏色
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID {self.id_list[i]} - {label}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

        cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 0, 255), 2)
        cv2.putText(frame, f"Car Count: {self.car_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            detected = self.detect_objects(frame)
            self.update_trackers(detected)
            self.draw(frame)

            cv2.imshow("YOLO + Kalman Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()



