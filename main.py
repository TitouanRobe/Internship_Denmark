from trackers import RoadSignsTracker
import cv2
from utils import (read_video,save_video)

def main():
    
    # Read Video 
    video_path = "./input/stop.mp4"
    # video_path = "traffic_sign.mp4"
    video_frames = read_video(video_path)
    
    # Detect Road Signs
    signs_tracker = RoadSignsTracker(model_path="./models/best.engine")
    signs_detections,class_detections = signs_tracker.detect_frames(video_frames)
    
    # Draw Bounding boxes 
    output_videos_frames = signs_tracker.draw_bboxes(video_frames, signs_detections, class_detections)
    save_video(output_videos_frames, output_video_path="output/video.avi")
    
    
    # Detect Vandalized signs 
    # damaged_detect = RoadSignsTracker(model_path="best_damaged2.pt")
    # damaged_detections,class_damaged = damaged_detect.detect_frames(video_frames)
    
    # # Draw Bounding boxes 
    # output_videos_frames = damaged_detect.draw_bboxes(video_frames, damaged_detections, class_damaged)
    # save_video(output_videos_frames, output_video_path="output/video_damaged.avi")

if __name__ == "__main__" : 
    
    main()