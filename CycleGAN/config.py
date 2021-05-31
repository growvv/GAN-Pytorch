import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Data/horse2zebra/horse2zebra/train"
VAL_DIR = "Data/horse2zebra/horse2zebra/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False    # 可以使用预训练模型
SAVE_MODEL = False
CHECKPOINT_GEN_H = "CycleGAN_weights/genh.pth.tar"
CHECKPOINT_GEN_Z = "CycleGAN_weights/genz.pth.tar"
CHECKPOINT_DISC_H = "CycleGAN_weights/disch.pth.tar"
CHECKPOINT_DISC_Z = "CycleGAN_weights/discz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)