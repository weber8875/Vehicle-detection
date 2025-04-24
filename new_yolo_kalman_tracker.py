import cv2
import onnxruntime as ort
import numpy as np
from scipy.optimize import linear_sum_assignment

# === ONNX 初始化 ===
session = ort.InferenceSession("yolov5s.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

with open("coco.names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# === 資料結構 ===
kalman_list = []

trackers = []
label_list = []

id_list = []
missed_frames = []
next_id = 0

MAX_MISSED = 10
DETECT_EVERY = 5
frame_id = 0

def create_kalman(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
    return kf

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

cap = cv2.VideoCapture("traffic4.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    h0, w0 = frame.shape[:2]
    detected = []

    # === 每 N 幀做 YOLO 偵測 ===
    should_detect = frame_id % DETECT_EVERY == 0
    if should_detect:
        img_resized = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, :]

        outputs = session.run([output_name], {input_name: img_input})[0]
        outputs = np.squeeze(outputs)

        for det in outputs:
            score = det[4]
            if score < 0.4:
                continue
            scores = det[5:]
            class_id = np.argmax(scores)
            label = class_names[class_id]
            confidence = scores[class_id]

            if confidence < 0.5 or label not in ["car", "motorcycle", "bicycle"]:
                continue

            cx, cy, w, h = det[:4]
            x = int((cx - w / 2) * w0 / 640)
            y = int((cy - h / 2) * h0 / 640)
            w = int(w * w0 / 640)
            h = int(h * h0 / 640)
            detected.append([x, y, w, h, label])

    # === Kalman 預測所有追蹤器 ===
    predicted_boxes = []
    for kf in kalman_list:
        pred = kf.predict()
        px = int(pred[0].item())
        py = int(pred[1].item())
        predicted_boxes.append((px - 20, py - 20, 40, 40))  # 模擬框

    # === 配對 ===
    cost_matrix = []
    for pb in predicted_boxes:
        row = []
        for db in detected:
            row.append(1 - iou(pb, db[:4]))
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

    # === 更新已匹配追蹤器 ===
    for tracker_idx, det_idx in matched_indices:
        x, y, w, h, label = detected[det_idx]
        cx = x + w // 2
        cy = y + h // 2
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        kalman_list[tracker_idx].correct(measurement)
        trackers[tracker_idx] = [x, y, w, h]
        label_list[tracker_idx] = label
        missed_frames[tracker_idx] = 0

    # === 累加未匹配 tracker 的 missed count ===
    for i in unmatched_trackers:
        missed_frames[i] += 1

    # === 移除長時間沒配對的 tracker ===
    # 要倒著刪除
    to_delete = [i for i, m in enumerate(missed_frames) if m > MAX_MISSED]
    for idx in sorted(to_delete, reverse=True):
        del kalman_list[idx]
        del trackers[idx]
        del id_list[idx]
        del missed_frames[idx]
        del label_list[idx]

    # === 新增新的 tracker ===
    for idx in unmatched_detections:
        x, y, w, h, label = detected[idx]
        cx = x + w // 2
        cy = y + h // 2
        kf = create_kalman(cx, cy)
        kalman_list.append(kf)
        trackers.append([x, y, w, h])
        id_list.append(next_id)
        missed_frames.append(0)
        label_list.append(label)
        next_id += 1

    # === 繪製畫面 ===
    for i, bbox in enumerate(trackers):
        x, y, w, h = bbox
        label = label_list[i]
        color = (0, 255, 0) if label == "car" else (0, 100, 255)  # 不同類別不同顏色
        pred = kalman_list[i].predict()
        px = int(pred[0].item())
        py = int(pred[1].item())

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)
        cv2.putText(frame, f"{label} ID {id_list[i]}", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("YOLO + Kalman + Optimized", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
