from trackers import RoadSignsTrackerPictures
import cv2

image_path = "./input/damaged_stop2.jpg"

def main():
    
    # Load Signs Detection Model 
    model_signs_detection = RoadSignsTrackerPictures(model_path="./models/best.pt")

    results_sign_detection = model_signs_detection.results(image_path)

    # Load Vandalized Detection Model
    vandalized_detection_model = RoadSignsTrackerPictures(model_path="./models/best_damage_detection_v2.pt")
    
    results_vandalized_detection = vandalized_detection_model.results(image_path)
    
    # Gather results in on picture
    model_signs_detection.analyse_frame(results_sign_detection, results_vandalized_detection, image_path, 0.5)

    
if __name__ == "__main__" : 
    
    main()