import os 
from tqdm import tqdm
from ultralytics import YOLO
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.video.source import get_video_frames_generator
from supervision.draw.color import ColorPalette
from supervision.video.sink import VideoSink, VideoInfo
from utils import *

config = load_yaml(os.path.join(os.getcwd(),"config","config.yaml"))
model = YOLO(config.MODEL)
model_classes_name = load_json(os.path.join(os.getcwd(),"config","yolov8_classes.json"))   # or model.model.names
video_info = VideoInfo.from_video_path(config.VIDEO_PATH)
generator = get_video_frames_generator(config.VIDEO_PATH)
box_annotator = BoxAnnotator(color = ColorPalette(),
                             thickness=config.ANNOTATIONS.THICKNESS, 
                             text_thickness=config.ANNOTATIONS.TEXT_THICKNESS, 
                             text_scale=config.ANNOTATIONS.TEXT_SCALE
                             )

with VideoSink(config.TARGET_VIDEO_PATH,video_info) as sink:
    for frame in tqdm(generator,total=video_info.total_frames):
        results = model(frame)[0]
        detections = Detections(
            xyxy =results.boxes.xyxy.cpu().numpy(),
            confidence = results.boxes.conf.cpu().numpy(),
            class_id = results.boxes.cls.cpu().numpy().astype(int)
            )
        labels = [f"{model_classes_name[class_id]} {confidence:0.2f}" for _,confidence,class_id,tracker_id in detections]
        frame = box_annotator.annotate(frame=frame,detections=detections,labels=labels)
        sink.write_frame(frame)