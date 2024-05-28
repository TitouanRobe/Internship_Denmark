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
# version des imports

#dataset_root = '../Road_Sign_Detection_in_Real_Time/'
dataset_root = './Damaged_Signs_Object_Detection/'
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
    
    print("\nAnnotations : \n")
    print(annotations)
    
    # Annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
    
    print("\nDetections : \n")
    print(detections)

    # Create label dictionary
    categories = TRAIN_DATASET.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    
    # Extract labels
    labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
    
    print("\n Labels : \n")
    print(labels)

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

import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch


class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path="./custom-model/",
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here:
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER
    
################################################################################################################################################################################

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

################################################################################################################################################################################

print(iter(TRAIN_DATALOADER).__dict__)
batch = next(iter(TRAIN_DATALOADER))

#print("\nici\n")

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

#print(outputs)

################################################################################################################################################################################
import torch
torch.cuda.empty_cache()
print(torch.cuda.is_available())

################################################################################################################################################################################

from pytorch_lightning import Trainer

# settings
MAX_EPOCHS = 200

trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=1)

trainer.fit(model)


################################################################################################################################################################################
# Save and load model

MODEL_PATH = 'custom-model'
model.model.save_pretrained(MODEL_PATH)

import torch

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading model
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(DEVICE)


################################################################################################################################################################################
# Inference and evaluation


CONFIDENCE_TRESHOLD = 0.5

# utils
categories = VAL_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()

# select random image
image_ids = VAL_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
print('Image #{}'.format(image_id))

# load image and annotatons
image = VAL_DATASET.coco.loadImgs(image_id)[0]
annotations = VAL_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(VAL_DATASET.root, image['file_name'])
image = cv2.imread(image_path)

# Annotate ground truth
detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
frame_ground_truth = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)




# Annotate detections
with torch.no_grad():

    # load image and predict
    inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
    outputs = model(**inputs)

    # post-process
    target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    results = image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=CONFIDENCE_TRESHOLD,
        target_sizes=target_sizes
    )[0]


    detections = sv.Detections.from_transformers(transformers_results=results)
    labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
    frame_detections = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)




# Combine both images side by side and display
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Display Ground Truth image
axs[0].imshow(cv2.cvtColor(frame_ground_truth, cv2.COLOR_BGR2RGB))
axs[0].axis('off')
axs[0].set_title('Ground Truth')
axs[0].set_aspect('auto')  # Adjust aspect ratio to fit the whole image

# Display Detections image
axs[1].imshow(cv2.cvtColor(frame_detections, cv2.COLOR_BGR2RGB))
axs[1].axis('off')
axs[1].set_title('Detections')
axs[1].set_aspect('auto')  # Adjust aspect ratio to fit the whole image

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm
from transformers import DetrImageProcessor
import numpy as np

# initialize evaluator with ground truth (gt)
evaluator = CocoEvaluator(coco_gt=VAL_DATASET.coco, iou_types=["bbox"])

processor = DetrImageProcessor.from_pretrained("C:/Users/robet/Downloads/Result_model/Result/custom-model/",
    use_safetensors=True)

print("Running evaluation...")
for idx, batch in enumerate(tqdm(VAL_DATALOADER)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(DEVICE)
    pixel_mask = batch["pixel_mask"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # turn into a list of dictionaries (one item for each example in the batch)
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)

    # provide to metric
    # metric expects a list of dictionaries, each item
    # containing image_id, category_id, bbox and score keys
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    predictions = prepare_for_coco_detection(predictions)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
