from ultralytics import YOLO 
import cv2
import sys
from PIL import Image
import os 


class RoadSignsTrackerPictures:
    
    def __init__(self, model_path):
        

        self.model = YOLO(model_path)
  
    def results(self, video_path):
        
        results = self.model.predict(video_path, conf=0.5, save=False)[0]
        
        id_name_dict = results.names
          
        player_dict = {}
        
        cp = 0 
        for box in results.boxes:
            #track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            confidence = round(box.conf[0].item(),2)

            player_dict[cp] = [result, object_cls_name, confidence]

            cp = cp + 1
        
        return player_dict
    
    def draw_bbox_image(self, image_parameters, video_path, color=(127, 255, 0)):
        
        frame = cv2.imread(video_path)
        
        for result in image_parameters.items() : 
            
            bbox = result[1][0]
            class_name = result[1][1]
            confidence = result[1][2]
            
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
            echelle = 0.6
            epaisseur = 1
            couleur_box = color
            couleur_texte = (255,255,255)
            text = class_name+" "+str(confidence)
    
            
            (taille_texte, _) = cv2.getTextSize(text, police, echelle, epaisseur)
            
            # Ajuster la largeur de la boîte englobante pour correspondre à la longueur du texte
            largeur_texte = taille_texte[0]
            largeur = largeur_texte

            # Dessiner le rectangle 


            # Dessiner la boîte englobante
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), couleur_box,3)
            
            
            cv2.rectangle(frame, (int(x1), int(y1) - taille_texte[1] - 5), (int(x1) + largeur , int(y1)), couleur_box, -1)

            # Afficher le nom de la classe en blanc à l'intérieur de la boîte englobante
            cv2.putText(frame, f"{text}", (int(bbox[0]),int(bbox[1] -5 )), cv2.FONT_HERSHEY_SIMPLEX, echelle, couleur_texte, epaisseur, cv2.LINE_AA)
            
            cv2.imwrite("./output/output1.jpg", frame)
    
    
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
    
