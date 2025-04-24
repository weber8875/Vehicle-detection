from SORT import ObjectTracker

if __name__ == '__main__':
    tracker = ObjectTracker(
        model_path = "yolov5s.onnx",
        class_name_path = "coco.names.txt",
        video_path = "traffic4.mp4"
    )
    tracker.run()