# Some basic setup:
# Detectron2 utils
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
import pandas as pd
# Image obtained
im = cv2.imread("input3.jpeg")
#cv2.imshow('ImageWindow',im)

# Create config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")

# Predictor created
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Map keypoints to image
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Get keypoints and create dataframe before passing to csv
output = outputs['instances'].pred_keypoints.to("cpu")
create_keypoint_dataset = pd.DataFrame(output.tolist())
#print(create_keypoint_dataset)
create_keypoint_dataset.to_csv('keypoints.csv')
#cv2.imshow('', out.get_image()[:, :, ::-1])
#cv2.waitKey(0)

# TODO: 
# Go through all the 'valid'  photos, extract skeleton data. Save to a file
# Build the next model, train it etc.
