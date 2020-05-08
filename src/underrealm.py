import glob
import json
import os
import random
from os.path import join, abspath, pardir

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.draw

from mrcnn import model as modellib, utils, visualize
from mrcnn.config import Config

PATH = join(abspath(join(abspath(__file__), pardir, pardir)), "data")
RGB_PATH = join(PATH, "rgb_images")
TRAIN_PATH = join(RGB_PATH, "train")
VAL_PATH = join(RGB_PATH, "val")

LOGS_PATH = join(abspath(join(abspath(__file__), pardir, pardir)), "logs")

COCO_WEIGHTS_PATH = os.path.join(join(abspath(join(abspath(__file__), pardir, pardir)), "model"), "mask_rcnn_coco.h5")


class UnderRealmConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "underrealm"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class UnderRealmDataset(utils.Dataset):
    def load_underrealm(self, dataset_dir, action):
        self.add_class("underrealm", 1, "underrealm")

        dataset_dir = join(dataset_dir, action)
        annotations = json.load(open(join(dataset_dir, action + ".json")))
        annotations = list(annotations.values())

        annotations = [a for a in annotations if a["regions"]]

        # Add images
        for a in annotations:
            if type(a["regions"]) is dict:
                polygons = [r["shape_attributes"] for r in a["regions"].values()]
            else:
                polygons = [r["shape_attributes"] for r in a["regions"]]

            image_path = os.path.join(dataset_dir, a["filename"])
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "underrealm",
                image_id=a["filename"],
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "underrealm":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "underrealm":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = UnderRealmDataset()
    dataset_train.load_underrealm(RGB_PATH, action="train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = UnderRealmDataset()
    dataset_val.load_underrealm(RGB_PATH, "val")
    dataset_val.prepare()

    # print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers="heads"
                )


def detect(model):
    # Get random image from validation dataset
    val_list = glob.glob(join(VAL_PATH, "*.png"))
    image_path = random.choice(val_list)
    # for image_path in val_list:
    print("Running on {}".format(image_path))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect objects
    r = model.detect([image], verbose=1)[0]

    class_name = ['BG', 'underrealm']

    depth_data = np.array(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
    depth_data = [[j * 0.01 for j in i] for i in depth_data]

    visualize.display_instances(image=image, boxes=r["rois"], masks=r["masks"], class_ids=r["class_ids"],
                                class_names=class_name, scores=r["scores"],
                                title="Predictions", depth_data=depth_data)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We"re treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


if __name__ == "__main__":
    config = UnderRealmConfig()
    config.display()
    config.BATCH_SIZE = 1
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 1

    # Training
    # model = modellib.MaskRCNN(mode="training", config=config,
    #                           model_dir=LOGS_PATH)
    # weights_path = COCO_WEIGHTS_PATH
    # model.load_weights(weights_path, by_name=True, exclude=[
    #     "mrcnn_class_logits", "mrcnn_bbox_fc",
    #     "mrcnn_bbox", "mrcnn_mask"])
    # train(model)
    # print("Training doneeeeee")

    # Detecting
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=LOGS_PATH)
    weights_path = model.find_last()
    model.load_weights(weights_path, by_name=True)
    detect(model)
    print("Detecting done!")
