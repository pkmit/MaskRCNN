"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import uuid
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath(os.getcwd())

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

DEFAULT_WEIGHT = os.path.join(ROOT_DIR, 'weight', 'mask_rcnn_roaddefect_0078.h5')
TEMP_PATH = os.path.join(ROOT_DIR, 'temp')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

############################################################
    #  Configurations
    ############################################################
class RoadDefectConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "RoadDefect"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class RoadDefectDataset(utils.Dataset):

    def load_road_defect(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Pothole", 1, "Pothole")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "Pothole",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):            
        image_info = self.image_info[image_id]
        if image_info["source"] != "Pothole":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Pothole":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)    

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    class_names = ['BG', 'Pothole']

    _uid = str(uuid.uuid4())[:8]
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Mask and Box image, return as byte stream        
        file_path = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], colors=generate_color(r['class_ids']), uid=_uid)
        # Save output
        shutil.copy2(file_path, ROOT_DIR)
        temp_folder_cleanup(_uid)
        # file_name = "splash_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
        # plt.imsave(file_name, plt.imread(img_buff))
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        uid = uuid.uuid4()
        file_path = "{}_{:%Y%m%dT%H%M%S}.avi".format(_uid, datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_path,
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, (width, height))
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], colors=generate_color(r['class_ids']), is_video=True, uid=_uid)
                splash = cv2.imread(splash)
                splash = cv2.resize(splash, (width, height))
                vwriter.write(splash)                
                count += 1
        vwriter.release()
        temp_folder_cleanup(_uid)
    print("Saved to ", file_path)

def temp_folder_cleanup(uid):
    p = os.path.join(ROOT_DIR, "temp", uid)
    print("Cleaning temp folder {}".format(p))
    shutil.rmtree(p)
    
def generate_color(class_ids):    
    id_color = [
        (0.0, 0.0, 0.0), #Background color
        (242 / 255.0, 5 / 255.0, 5 / 255.0) #Pothole
    ]

    colors = list()

    for id in class_ids:
        colors.append(id_color[id])

    return colors

class PKMAI_BACKBONE:        
    def __init__(self):  
        class InferenceConfig(RoadDefectConfig):            
            GPU_COUNT = 1 # Set batch size to 1 since we'll be running inference on
            IMAGES_PER_GPU = 1 # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

        config = InferenceConfig()
        config.display()
        
        weights_path = DEFAULT_WEIGHT
        self.model = modellib.MaskRCNN(mode="inference", config=config, model_dir=LOG_DIR)
        self.model.load_weights(weights_path, by_name=True)
            
    def prediction(self, img_path):
        detect_and_color_splash(self.model, image_path=img_path, video_path=None)