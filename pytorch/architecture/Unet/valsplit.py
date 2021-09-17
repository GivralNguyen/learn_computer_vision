import os
import shutil 

train_dir = "train"
train_masks_dir = "train_masks"
val_dir = "val_images"
val_masks_dir = "val_masks"

print("Number of training images is ",len(os.listdir(train_dir)))
print("Number of training mask is ",len(os.listdir(train_masks_dir)))

sorted_train_dir = sorted(os.listdir(train_dir))
sorted_train_mask_dir = sorted(os.listdir(train_masks_dir))

val_split_ratio = 0.1
for img_id, file_name in enumerate(sorted_train_dir):
    if (img_id % (1/val_split_ratio)==1):
        shutil.move("train/"+file_name, "val_images/"+file_name)
    
for img_id, file_name in enumerate(sorted_train_mask_dir):
    if (img_id % (1/val_split_ratio)==1):
        shutil.move("train_masks/"+file_name, "val_masks/"+file_name)