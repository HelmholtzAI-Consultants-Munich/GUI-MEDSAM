import os
os.environ['HYDRA_FULL_ERROR']='1'
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz

from dataclasses import dataclass

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

sam2predictor = build_sam2_video_predictor_npz(
    "configs/sam2.1_hiera_t512.yaml",
    "checkpoints/MedSAM2_latest.pt",
    device)

@dataclass
class Point:
    x: float
    y: float
    z: float = 0

@dataclass
class ClicksInfo:
    positive: list[Point]
    negative: list[Point]

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

@torch.inference_mode()
def infer_3d(img_3D: np.ndarray, clicks_list: list[ClicksInfo]):
    """
    Run inference on a 3D numpy image with multiple objects defined by clicks.
    
    Args:
        img_3D: Input image as numpy array with shape (D, H, W) in range [0, 255]
        clicks_list: List of ClicksInfo objects, one per object to segment
        
    Returns:
        List of 3D segmentation masks (numpy arrays), one per object
    """
    assert np.max(img_3D) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D)}'
    
    # Handle both 2D and 3D images
    if img_3D.ndim == 2:
        img_3D = img_3D[np.newaxis, :, :]  # Add depth dimension for 2D images
    
    img_3D_ori = img_3D
    
    # resize image to 512x512 and normalize
    video_height, video_width = img_3D_ori.shape[1:3]
    if video_height != 512 or video_width != 512:
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512) #1024) #d, 3, 1024, 1024
    else:
        img_resized = img_3D_ori[:,None].repeat(3, axis=1) # d, 3, 1024, 1024
    
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized.astype(np.float32)).to(device)
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].to(device)
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].to(device)
    img_resized -= img_mean
    img_resized /= img_std
    
    # List to store segmentation masks for each object
    all_segmentations = []
    
    # iterate over each object defined by clicks
    for obj_idx, clicks_info in enumerate(clicks_list):
        print(f'Processing object {obj_idx + 1}/{len(clicks_list)}')
        
        # Initialize segmentation mask for this object
        segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)
        
        if len(clicks_info.positive) == 0 and len(clicks_info.negative) == 0:
            print(f'No clicks for object {obj_idx}, skipping...')
            all_segmentations.append(segs_3D)
            continue
        
        # Group clicks by z-index
        clicks_by_frame = {}
        for point in clicks_info.positive:
            z = int(point.z)
            if z not in clicks_by_frame:
                clicks_by_frame[z] = {'points': [], 'labels': []}
            clicks_by_frame[z]['points'].append([point.x, point.y])
            clicks_by_frame[z]['labels'].append(1)
        
        for point in clicks_info.negative:
            z = int(point.z)
            if z not in clicks_by_frame:
                clicks_by_frame[z] = {'points': [], 'labels': []}
            clicks_by_frame[z]['points'].append([point.x, point.y])
            clicks_by_frame[z]['labels'].append(0)
        
        # add prompt to initialize the predictor
        with torch.inference_mode():
            inference_state = sam2predictor.init_state(img_resized, video_height, video_width)
            
            # Add clicks for each frame
            for frame_idx, click_data in clicks_by_frame.items():
                points = np.array(click_data['points'], dtype=np.float32)
                labels = np.array(click_data['labels'], dtype=np.int32)
                print("Adding clicks: frame", frame_idx, points, labels)
                _, out_obj_ids, out_mask_logits = sam2predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )

                print(np.sum(out_mask_logits.detach().cpu().numpy() > 0))

            min_marked_frame = min(clicks_by_frame.keys())
            max_marked_frame = max(clicks_by_frame.keys())

            # Propagate forward from the first frame
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2predictor.propagate_in_video(
                inference_state, start_frame_idx=min_marked_frame, reverse=False
            ):
                segs_3D[out_frame_idx] = out_mask_logits.detach().cpu().numpy() > 0

            for out_frame_idx, out_obj_ids, out_mask_logits in sam2predictor.propagate_in_video(
                inference_state, start_frame_idx=max_marked_frame, reverse=True
            ):
                segs_3D[out_frame_idx] = out_mask_logits.detach().cpu().numpy() > 0
            
            sam2predictor.reset_state(inference_state)
        
        all_segmentations.append(segs_3D)
    
    return all_segmentations
