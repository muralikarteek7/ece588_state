import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from segment_anything import sam_model_registry, SamPredictor

# Load and preprocess the image
image = cv2.imread('scans/color1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM model
sam_checkpoint = "/home/karteek/ece588/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

# Define bounding box
input_box = np.array([500, 510, 650, 600])  # Format: [x_min, y_min, x_max, y_max]

# Perform segmentation using bounding box
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

# Get the best mask
best_mask = masks[0]

# Visualize the mask and bounding box
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.imshow(best_mask, alpha=0.5, cmap='jet')  # Overlay the mask

# Draw bounding box
x_min, y_min, x_max, y_max = input_box
rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                     edgecolor='yellow', linewidth=2, fill=False, label="Bounding Box")
plt.gca().add_patch(rect)

plt.legend()
plt.axis('off')

# Save the mask
mask_filename = "segmentation_mask.png"
cv2.imwrite(mask_filename, (best_mask * 255).astype(np.uint8))

# Show the result
plt.show()
