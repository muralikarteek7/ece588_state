import cv2
import open3d as o3d
import numpy as np
import rasterio
import matplotlib.pyplot as plt

color_img_path = "scans/color1.png"
depth_img_path = "scans/depth1.tif"

color_img = cv2.imread(color_img_path)
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)

lower_bound = np.array([98, 43, 0])
upper_bound = np.array([118, 143, 94])

mask = cv2.inRange(hsv, lower_bound, upper_bound)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea) if contours else None

box_mask = np.zeros_like(mask, dtype=np.uint8)
if largest_contour is not None:
    cv2.drawContours(box_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

plt.figure(figsize=(6, 4))
plt.imshow(box_mask, cmap='gray')
plt.title("Detected Closest Box Mask")
plt.axis("off")
plt.show()

with rasterio.open(depth_img_path) as src:
    depth_array = src.read(1)

masked_depth = np.where(box_mask == 255, depth_array, 0)

fx, fy = 684.8333, 684.6097
cx, cy = 573.3711, 363.7009

points = []
colors = []
h, w = depth_array.shape

for y in range(h):
    for x in range(w):
        d = masked_depth[y, x]
        if d > 0:
            z = d * 0.1
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points.append([X, Y, z])
            colors.append(color_img[y, x] / 255.0)

pcd = o3d.geometry.PointCloud()
if points:
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

o3d.visualization.draw_geometries([pcd])
