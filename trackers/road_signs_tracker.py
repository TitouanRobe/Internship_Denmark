from ultralytics import YOLO 
import cv2
import sys
from PIL import Image
import os 


class RoadSignsTracker:
    
    def __init__(self,model_path):
        
        self.model = YOLO(model_path)
        
        
    def detect_frame(self,frame):
        results = self.model.predict(frame, conf=0.5)[0]
        id_name_dict = results.names

        player_dict = {}
        class_names = {}
        cp = 0 
        for box in results.boxes:
            #track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            
            print(object_cls_id)
            print(object_cls_name)
            player_dict[cp] = result
            class_names[cp] = object_cls_name
            
            cp = cp + 1
        
        return player_dict,class_names
    
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []
        class_names_detections= []
        for frame in frames:
            player_dict, class_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
            class_names_detections.append(class_dict)
        
       
        return player_detections, class_names_detections
    
    
    def draw_bbox(self,bbox,class_name,frame):
        
        x1, y1, x2, y2 = bbox
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours and find the largest one (assuming the sign is the largest object in the image)
        sign_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour with a polygon
        approx = cv2.approxPolyDP(sign_contour, 0.01 * cv2.arcLength(sign_contour, True), True)

        police = cv2.FONT_HERSHEY_SIMPLEX
        echelle = 1.5
        epaisseur = 2
        couleur_box = (127, 255, 0)
        couleur_texte = (255,255,255)
        
        (taille_texte, _) = cv2.getTextSize(class_name, police, echelle, epaisseur)
        
        # Ajuster la largeur de la boîte englobante pour correspondre à la longueur du texte
        largeur_texte = taille_texte[0]
        largeur = largeur_texte

        # Dessiner le rectangle 


        # Dessiner la boîte englobante
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), couleur_box, thickness=6)
        
        
        cv2.rectangle(frame, (int(x1), int(y1) - taille_texte[1] - 15), (int(x1) + largeur , int(y1)), couleur_box, -1)

        # Afficher le nom de la classe en blanc à l'intérieur de la boîte englobante
        cv2.putText(frame, f"{class_name}", (int(bbox[0]),int(bbox[1] -10 )), cv2.FONT_HERSHEY_SIMPLEX, echelle, couleur_texte, epaisseur, cv2.LINE_AA)


    def draw_bboxes(self,video_frames, signs_detections, class_names):
        output_video_frames = []

        for frame, player_dict, class_name in zip(video_frames, signs_detections, class_names):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                    x1, y1, x2, y2 = bbox
                    
                    self.draw_bbox(bbox, class_name[track_id], frame)
                    # cv2.putText(frame, f"{class_name[track_id]}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
            
    
    def calculate_iou(self, box1, box2):

        #Calculate intersection over union (IoU) between two bounding boxes.
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area_box1 + area_box2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou
    
    
    def boxes_match(self,box1, box2, iou_threshold=0.3):
        
        # Check if two bounding boxes match based on IoU threshold.
        iou = self.calculate_iou(box1, box2)
        
        return iou >= iou_threshold
    
    def analyse_frame(self, frame_signs_detections, frame_vandalized_detections, image_path, threshold):
        

        for i in range (len(frame_signs_detections)): 
            
            color_box = (127, 255, 0)
    
            bbox_signs = frame_signs_detections[i][0]
            
            # if frame_vandalized_detections is not empty
            if frame_vandalized_detections :
                
                bbox_vandalization = frame_vandalized_detections[i][0]
                
                confidence_Signs_detection = frame_signs_detections[i][2]
                confidence_Vandalized_detection = frame_vandalized_detections[i][2]
                
    

                if confidence_Vandalized_detection > threshold : 

                    if self.boxes_match(bbox_signs,bbox_vandalization): 
                        
                        # Name changes for Damaged sign 
                        frame_signs_detections[i][1] = "Damaged "+frame_signs_detections[i][1]
                        # Confiance du deuxième modèle pris comme référence
                        frame_signs_detections[i][2] = frame_vandalized_detections[i][2]
                        
                        color_box = (60, 20, 220)
                else : 
                    
                        frame_signs_detections[len(frame_signs_detections)] = frame_vandalized_detections[0]
                        print(frame_signs_detections)
                
            self.draw_bbox_image(frame_signs_detections, image_path, color_box)
    

