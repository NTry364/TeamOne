
from ultralytics import YOLO

# Load a model
model = YOLO("./yolov8n.pt")  # load a pretrained model (recommended for training)
# Run batched inference on a list of images

results = model("DO-XE-35/test/images/bai15_jpg.rf.8beebbd6bac2541cfe2a63c1968f84bc.jpg")  # predict on an image
# Process results list
for result in results:
    ##boxes = result.boxes  # Boxes object for bounding box outputs
    ##masks = result.masks  # Masks object for segmentation masks outputs
    ##keypoints = result.keypoints  # Keypoints object for pose outputs
   ##probs = result.probs  # Probs object for classification outputs
    ##obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result_2.jpg")  # save to disk


