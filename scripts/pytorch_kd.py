import torch 
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.nn.functional as F
import torch
import supervision as sv
import transformers
import pytorch_lightning
import os
import random
import cv2
import numpy as np
import timm
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import torchvision.transforms as transforms
# version des imports

dataset_root = './Road_Sign_Detection_in_Real_Time/'
#dataset_root = './Damaged_Signs_Object_Detection/'
images_folder = 'images'  # Subfolder containing images

ANNOTATION_FILE_NAME = "annotations.json"
TRAIN_DIRECTORY = os.path.join(dataset_root, "Train", images_folder)
VAL_DIRECTORY = os.path.join(dataset_root, "Validation", images_folder)
TEST_DIRECTORY = os.path.join(dataset_root, "test", images_folder)

print(dataset_root)
print(TRAIN_DIRECTORY)
print(VAL_DIRECTORY)

import torch
from transformers import DetrImageProcessor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


import torchvision
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        annotation_directory_path: str,
        image_processor,
        train: bool = True
    ):
        annotation_file_path = os.path.join(annotation_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    
TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, annotation_directory_path=dataset_root + "Train", image_processor=image_processor, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, annotation_directory_path=dataset_root + "Validation", image_processor=image_processor, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, annotation_directory_path=dataset_root + "test", image_processor=image_processor, train=False)
print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples: ", len(TEST_DATASET))

################################################################################################################################################################################

# select random image
image_ids = TRAIN_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
image_info = TRAIN_DATASET.coco.loadImgs(image_id)[0]
print('Image #{}: {}'.format(image_id, image_info['file_name']))  # Print image file name

# Construct the correct image file path
image_path = os.path.join(TRAIN_DATASET.root, image_info['file_name'])

# Check if the image file exists
if os.path.exists(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Load annotations
    annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
    
    #print("\nAnnotations : \n")
    #print(annotations)
    
    # Annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
    
    #print("\nDetections : \n")
    #print(detections)

    # Create label dictionary
    categories = TRAIN_DATASET.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    
    # Extract labels
    labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
    
    #print("\n Labels : \n")
    #print(labels)

    # Create box annotator
    box_annotator = sv.BoxAnnotator()

    # Annotate the image
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    # Display the annotated image with Matplotlib
    sv.show_frame_in_notebook(image, (8, 8))

    image_name = image_path.split("/")[-1]
    
    path_image_save = "./results/" + image_name
    

    #cv2.imwrite(path_image_save, frame)
    
    #print("Chemin de l'image créée:", path_image_save)
    #cv2.imwrite()
    
else:
    print("Image file does not exist:", image_path)
    # Handle the case where the image file does not exist
    
    
################################################################################################################################################################################
    
# In some cases, for instance, when fine-tuning DETR, the model applies scale augmentation at training time.
# This may cause images to be different sizes in a batch. You can use DetrImageProcessor.pad() from DetrImageProcessor and define a custom collate_fn to batch images together.
from torch.utils.data import DataLoader

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }
    
TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)

################################################################################################################################################################################

def kd(teacher, student, DATALOADER, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    print("Start...")
    i=0
    for batch in DATALOADER : 
        if i == 0 :
            
            inputs, labels = batch["pixel_values"], batch["labels"]
            
            pixels_values, pixel_mask = batch["pixel_values"], batch["pixel_mask"]
            # inputs = inputs.to(device)
            # teacher = teacher.to(device)
            # student = student.to(device)
            
            # # Vérifier la taille de chaque tensor pour s'assurer qu'ils ont tous la même taille
            # tensor_sizes = {key: value.size() for key, value in labels[0].items()}
            # if not all(value.size() == tensor_sizes[key] for key, value in labels[1].items()):
            #     raise ValueError("Les tensors n'ont pas la même taille")

            # # Concaténer les valeurs de chaque clé dans une liste de tensors
            # concatenated_tensors = {key: torch.cat([d[key] for d in labels], dim=0) for key in keys}

            # # Créer un tensor unique en empilant les tensors concaténés le long d'une nouvelle dimension
            # labels_tensor = torch.stack([concatenated_tensors[key] for key in keys], dim=1)

            # # Afficher le tensor résultant
            # print(labels_tensor)
            # sys.exit()
            
            optimizer.zero_grad()

            print("Suite...")
            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)["logits"]
                outputs = teacher(pixel_values=pixels_values, pixel_mask=pixel_mask)
            
            # print("OUTPUTS")
            # print(outputs["logits"])
            
            # print(outputs["logits"].shape)
            #print(teacher_logits)

            # Forward pass with the student model
            student_logits = student(pixels_values)["logits"]
            
            linear_layer = nn.Linear(1000, 100 * 75)

            # Appliquer la couche linéaire aux logits de l'étudiant
            adjusted_logits = linear_layer(student_logits)

            # Redimensionner les logits pour obtenir la forme [4, 100, 75]
            adjusted_logits = torch.reshape(adjusted_logits, (4, 100, 75))

            print(labels)
            print(len(labels))
            
            labels_cat = []
            for label in labels : 
                
                labels_cat.append(label["class_labels"])
            labels_cat = torch.cat(labels_cat, dim=0)
                
            # print(labels_cat)
            # print(labels_cat.shape)
            
            # print("INPUT SHAPE")
            # print(inputs.shape)
            
            # print(student_logits.shape)
            # print(adjusted_logits.shape)
            print("OUTPUTS TEACHER")
            #print(outputs["logits"])
            print(outputs["logits"].shape)
            
            print("OUTPUTS STUDENT")
            #print(student_logits)
            print(student_logits.shape)
            
            print("OUTPUTS STUDENTS ADJUSTED")
            #print(adjusted_logits)
            print(adjusted_logits.shape)
            

            
            student_logits = adjusted_logits
            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
            
            
            # print("SOFT TARGETS")
            # #print(soft_targets)
            # print(soft_targets.shape)
            
            # print("SOFT PROBS")
            # #print(soft_prob)
            # print(soft_prob.shape)            

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
            print(soft_targets_loss)
            
            print("TEACHER LOGITs")
            print(teacher_logits.shape)

            print("STUDENT LOGITs")
            print(student_logits.shape)

            print("LABELS")
            print(labels_cat.shape)
            
            tensor_redimensionne = labels_cat.view(4, 75)
            
            print("REDIMENSIONNE")
            print(tensor_redimensionne)
            print(tensor_redimensionne.shape)
            
            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels_cat)
            sys.exit()
            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        i=i+1
        break
    print("ici")
    
    
from transformers import DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained(
    "C:/Users/robet/Downloads/Result_model/Result/custom-model/",
    use_safetensors=True)

from transformers import AutoImageProcessor, AutoModelForImageClassification

student_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

total_params_deep = "{:,}".format(sum(p.numel() for p in model.parameters()))
print(f"DeepNN parameters: {total_params_deep}")

total_params_deep = "{:,}".format(sum(p.numel() for p in student_model.parameters()))
print(f"DeepNN parameters: {total_params_deep}")

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            
            batch
            pixel_values, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)


            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
#train_knowledge_distillation(teacher=model, student=student_model, train_loader=TRAIN_DATALOADER, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=DEVICE)

# # Compare the student test accuracy with and without the teacher, after distillation
# print("cic")

kd(model, student_model, TRAIN_DATALOADER, 10, 0.001, 2, 0.25, 0.75, DEVICE)

# for batch in TRAIN_DATALOADER : 
    
#     print(batch)
    
#     labels = batch["labels"]
#     break


# print(labels)