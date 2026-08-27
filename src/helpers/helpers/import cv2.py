import cv2

# Load the image in grayscale
img = cv2.imread("/home/chirag/Pictures/Screenshots/coep.png", cv2.IMREAD_GRAYSCALE)

# Isolate the white paths. Pixels brighter than 240 become 255 (white). 
# Everything else (river, buildings, grass) becomes 0 (black).
_, binary_map = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

# Save as PGM for the ROS map server
cv2.imwrite("coep_campus_map.pgm", binary_map)
print("Occupancy grid generated.")