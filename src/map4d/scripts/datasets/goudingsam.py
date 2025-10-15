import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/data/workspace/zhengguangting/map4d/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/data/workspace/zhengguangting/map4d/GroundingDINO/weights/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/data/workspace/zhengguangting/map4d/segment-anything/sam_vit_h_4b8939.pth"

def segment_sky(image_paths):
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    mask_list = []
    for i in range(len(image_paths)):
        # Predict classes and hyper-param for GroundingDINO
        SOURCE_IMAGE_PATH = image_paths[i]
        CLASSES = ["sky"]
        BOX_THRESHOLD = 0.5
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8

        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # save the annotated grounding dino image
        cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)


        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        if detections.confidence.shape[0] == 0:
            mask_list.append(np.zeros((image.shape[0],image.shape[1])).astype(bool))
            continue
        print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)


        # convert detections to masks
        mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        result_bool_array = np.any(mask, axis=0)
        mask_list.append(result_bool_array)
        detections.mask = mask
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # save the annotated grounded-sam image
        cv2.imwrite("grounded_sam_annotated_image"+str(i)+".jpg", annotated_image)
    return mask_list

def segment_txt1(images, txt, box_threshold=0.2, text_threshold=0.2, nms_threshold=0.8):
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    # print(111)
    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    # print(111)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    # print(111)
    mask_list = []
    for i in range(len(images)):
        # Predict classes and hyper-param for GroundingDINO
        image = images[i]
        # image = (image * 255).astype(np.uint8)
        if isinstance(txt, list):
            CLASSES = txt
        else:
            CLASSES = [txt]
        BOX_THRESHOLD = box_threshold
        TEXT_THRESHOLD = text_threshold
        NMS_THRESHOLD = nms_threshold

        # load image
        # image = cv2.imread(SOURCE_IMAGE_PATH)
        # print(222)
        # detect objects
        print(CLASSES)
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        # print(333)
        # print(detections)
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{confidence:0.2f}" 
            for _, _, confidence, class_id, _, _
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        # print(444)
        # save the annotated grounding dino image
        cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)


        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        if detections.confidence.shape[0] == 0:
            mask_list.append(np.zeros((image.shape[0],image.shape[1])).astype(bool))
            continue
        print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)


        # convert detections to masks
        mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        result_bool_array = np.any(mask, axis=0)
        mask_list.append(result_bool_array)
        
    return mask_list