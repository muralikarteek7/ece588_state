import cv2
import numpy as np

# Load the RGB image and mask
color_path = "./scans/color1.png"
mask_path = "segmentation_mask.png"

color_img = cv2.imread(color_path)
mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Grayscale mask

# Overlay mask on color image for better visualization
overlay = cv2.addWeighted(color_img, 0.7, cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR), 0.3, 0)

# List to store clicked points
clicked_points_2D = []

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points_2D.append((x, y))
        print(f"Clicked Pixel: {x}, {y}")

# Display the image
cv2.imshow("Click on Image (Press 'q' to exit)", overlay)
cv2.setMouseCallback("Click on Image (Press 'q' to exit)", click_event)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

cv2.destroyAllWindows()

print("\nCollected 2D Points:", clicked_points_2D)
