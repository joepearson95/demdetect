# Some basic setup:
# Detectron2 utils
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os.path
import cv2
import numpy as np
import pandas as pd
import glob


# Add column names dynamically
col_names = []
for x in range(17):
    col_names.append("point_" + str(x) + "_x")
    col_names.append("point_" + str(x) + "_y")
    col_names.append("point_" + str(x) + "_score")

cols = pd.DataFrame(col_names)
cols = cols.T
create_keypoint_dataset = pd.DataFrame()
create_keypoint_dataset = create_keypoint_dataset.append(col_names).T

# Loop through all 'valid' photos
for filename in glob.iglob('input*.jpeg', recursive=True):
    # Image obtained
    im = cv2.imread(filename)
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
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Obtain just the keypoints and make a flat list for the dataset
    output = outputs['instances'].pred_keypoints.to("cpu")
    listed = output.tolist()
    flattened =  sum(sum(listed,[]), [])
    create_keypoint_dataset = create_keypoint_dataset.append(pd.DataFrame(flattened).T)
    
# Comment out when testing
if os.path.exists("keypoints.csv") == False:
    create_keypoint_dataset.to_csv('keypoints.csv', mode='a',index=False, header=False)
else:
    print("File already exists.")

#cv2.imshow('', out.get_image()[:, :, ::-1])
#cv2.waitKey(0)

# TODO: 
# Go through all the 'valid'  photos, extract skeleton data. Save to a file
# Build the next model, train it etc.
