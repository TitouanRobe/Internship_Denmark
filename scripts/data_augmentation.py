#This is the train.py file

from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr

def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A         

            # Insert required transformation here
            T = [
                A.RandomRain(p=0.1,slant_lower=-10, slant_upper=10,drop_length=20, drop_width=1, drop_color=(200, 200, 200),blur_value=5, brightness_coefficient=0.9, rain_type=None),
                A.Rotate(limit = 10, p=0.5),
                A.Blur(p=0.1),
                A.MedianBlur(p=0.1),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

Albumentations.__init__ = __init__

model = YOLO('yolov8n.yaml').load('yolov8n.pt')

results = model.train(data='coco128.yaml', epochs=100, imgsz=640, augment = True)