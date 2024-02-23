import folder_paths
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import os

folder_paths.folder_names_and_paths["yolov8"] = (
    [os.path.join(folder_paths.models_dir, "yolov8")],
    folder_paths.supported_pt_extensions,
)


class Yolov8DetectionNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list("yolov8"),),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    FUNCTION = "detect"
    CATEGORY = "yolov8"

    def detect(self, image, model_name):
        # Convert tensor to numpy array and then to PIL Image
        image_tensor = image
        image_np = image_tensor.cpu().numpy()  # Change from CxHxW to HxWxC for Pillow
        image = Image.fromarray(
            (image_np.squeeze(0) * 255).astype(np.uint8)
        )  # Convert float [0,1] tensor to uint8 image

        print(
            f'model_path: {os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}'
        )
        model = YOLO(
            f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}'
        )  # load a custom model
        results = model(image)

        # TODO load masks
        # masks = results[0].masks

        im_array = results[0].plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

        image_tensor_out = torch.tensor(
            np.array(im).astype(np.float32) / 255.0
        )  # Convert back to CxHxW
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)

        return (
            image_tensor_out,
            {"classify": [r.boxes.cls.tolist()[0] for r in results]},
        )


class Yolov8SegNode:
    def __init__(self) -> None: ...

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list("yolov8"),),
                "class_id": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    FUNCTION = "seg"
    CATEGORY = "yolov8"

    def seg(self, image, model_name, class_id):
        model = YOLO(
            f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}'
        )  # load a custom model

        # Convert tensor to numpy array and then to PIL Image
        tensor_image = image
        img_list = []
        mask_list = []
        for image_cur in tensor_image:
            image_np = image_cur.cpu().numpy()  # Change from CxHxW to HxWxC for Pillow
            image = Image.fromarray(
                (image_np.squeeze(0) * 255).astype(np.uint8)
            )  # Convert float [0,1] tensor to uint8 image

            print(
                f'model_path: {os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}'
            )

            results = model(image)

            # get array results
            masks = results[0].masks.data
            boxes = results[0].boxes.data
            # extract classes
            clss = boxes[:, 5]
            # get indices of results where class is 0 (people in COCO)
            people_indices = torch.where(clss == class_id)
            # use these indices to extract the relevant masks
            people_masks = masks[people_indices]
            # scale for visualizing results
            people_mask = torch.any(people_masks, dim=0).int() * 255

            im_array = results[0].plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

            image_tensor_out = torch.tensor(
                np.array(im).astype(np.float32) / 255.0
            )  # Convert back to CxHxW
            img_list.append(image_tensor_out)
            mask_list.append(people_mask)

        return (torch.stack(img_list, dim=0), torch.stack(mask_list, dim=0))


NODE_CLASS_MAPPINGS = {
    "Yolov8Detection": Yolov8DetectionNode,
    "Yolov8Segmentation": Yolov8SegNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yolov8Detection": "detection",
    "Yolov8Segmentation": "seg",
}
