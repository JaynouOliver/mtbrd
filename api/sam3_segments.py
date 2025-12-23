import os
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from voyage_embedding import embed_image_path


load_dotenv()

def sam3_segment_image(image_path: str) -> dict:
    """
    Runs the SAM3 workflow on the given image path, returns result.
    """
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.getenv("ROBOFLOW_API_KEY")
    )
    result = client.run_workflow(
        workspace_name="mattoboard-alsau",
        workflow_id="sam3-with-prompts",
        images={
            "image": image_path  # Path to your image file
        },
        parameters={
            "prompts": ["furniture", "walls", "floors"]
        },
        use_cache=True
    )
    return result


def process_segments_and_save(
    result: dict,
    original_image_path: str,
    output_dir: str = "api/sam3_segments/",
    background_color: str = "white",
    use_transparent: bool = False
) -> List[str]:
    """
    Process SAM3 segmentation results: crop images to segments and mask backgrounds.
    
    Args:
        result: SAM3 workflow result dictionary containing predictions
        original_image_path: Path to the original image file
        output_dir: Directory to save processed segment images
        background_color: Background color for masked areas ("white" or "transparent")
        use_transparent: If True, use transparent background instead of white
        
    Returns:
        List of paths to saved segment images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    original_image = Image.open(original_image_path).convert("RGB")
    original_array = np.array(original_image)
    height, width = original_array.shape[:2]
    
    image_name = Path(original_image_path).stem
    saved_paths = []
    
    if isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        if isinstance(first_item, dict) and "sam" in first_item:
            predictions = first_item["sam"].get("predictions", [])
        else:
            predictions = result
    elif isinstance(result, dict):
        if "sam" in result:
            predictions = result["sam"].get("predictions", [])
        else:
            predictions = result.get("predictions", [])
            if not predictions:
                predictions = result.get("results", [])
            if not predictions:
                predictions = result.get("data", [])
            if not predictions:
                predictions = result.get("outputs", [])
    else:
        raise ValueError(f"Unexpected result type: {type(result)}")
    
    if not predictions:
        raise ValueError("No predictions found in SAM3 result")
    
    for idx, prediction in enumerate(predictions):
        mask = None
        bbox = None
        
        if "rle_mask" in prediction:
            rle_data = prediction["rle_mask"]
            if isinstance(rle_data, dict):
                rle_data["size"] = [height, width]
                mask = mask_utils.decode(rle_data)
        
        if mask is None and "mask" in prediction:
            mask_data = prediction["mask"]
            if isinstance(mask_data, dict):
                if "rle" in mask_data:
                    rle = mask_data["rle"]
                    if isinstance(rle, dict):
                        rle["size"] = [height, width]
                    mask = mask_utils.decode(rle)
                elif "segmentation" in mask_data:
                    mask = np.array(mask_data["segmentation"], dtype=np.uint8)
            elif isinstance(mask_data, (list, np.ndarray)):
                mask = np.array(mask_data, dtype=np.uint8)
                if mask.ndim == 2:
                    pass
                elif mask.ndim == 3:
                    mask = mask[:, :, 0] if mask.shape[2] == 1 else mask.max(axis=2)
        
        if mask is None and "segmentation" in prediction:
            seg = prediction["segmentation"]
            if isinstance(seg, dict) and "rle" in seg:
                rle = seg["rle"]
                if isinstance(rle, dict):
                    rle["size"] = [height, width]
                mask = mask_utils.decode(rle)
            elif isinstance(seg, (list, np.ndarray)):
                mask = np.array(seg, dtype=np.uint8)
        
        if mask is None:
            continue
        
        if mask.shape != (height, width):
            mask = Image.fromarray(mask).resize((width, height), Image.NEAREST)
            mask = np.array(mask)
        
        mask_binary = (mask > 0).astype(np.uint8)
        
        if bbox is None and "bbox" in prediction:
            bbox = prediction["bbox"]
        elif bbox is None and all(key in prediction for key in ["x", "y", "width", "height"]):
            x_center = prediction["x"]
            y_center = prediction["y"]
            w = prediction["width"]
            h = prediction["height"]
            x_min = int(x_center - w / 2)
            y_min = int(y_center - h / 2)
            x_max = int(x_center + w / 2)
            y_max = int(y_center + h / 2)
            bbox = [x_min, y_min, x_max, y_max]
        elif bbox is None:
            coords = np.where(mask_binary > 0)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                bbox = [x_min, y_min, x_max, y_max]
            else:
                continue
        
        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = map(int, bbox)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            cropped_image = original_array[y_min:y_max, x_min:x_max]
            cropped_mask = mask_binary[y_min:y_max, x_min:x_max]
            
            if use_transparent or background_color == "transparent":
                masked_image = np.zeros(
                    (cropped_image.shape[0], cropped_image.shape[1], 4),
                    dtype=np.uint8
                )
                masked_image[:, :, :3] = cropped_image
                masked_image[:, :, 3] = (cropped_mask * 255).astype(np.uint8)
                pil_image = Image.fromarray(masked_image)
            else:
                masked_image = cropped_image.copy()
                white_bg = np.ones_like(cropped_image) * 255
                mask_3d = cropped_mask[:, :, np.newaxis]
                masked_image = masked_image * mask_3d + white_bg * (1 - mask_3d)
                pil_image = Image.fromarray(masked_image.astype(np.uint8))
            
            class_name = prediction.get("class", prediction.get("label", "segment"))
            filename = f"{image_name}_{class_name}_{idx}.png"
            output_path = os.path.join(output_dir, filename)
            pil_image.save(output_path)
            saved_paths.append(output_path)
    
    return saved_paths


def segment_and_process_image(
    image_path: str,
    output_dir: str = "api/sam3_segments/",
    background_color: str = "white",
    use_transparent: bool = False
) -> List[str]:
    """
    Complete workflow: segment image with SAM3, then process and save segments.
    
    Args:
        image_path: Path to the image file to segment
        output_dir: Directory to save processed segment images
        background_color: Background color for masked areas ("white" or "transparent")
        use_transparent: If True, use transparent background instead of white
        
    Returns:
        List of paths to saved segment images
    """
    result = sam3_segment_image(image_path)
    return process_segments_and_save(
        result,
        image_path,
        output_dir,
        background_color,
        use_transparent
    )


def embed_segment_images(
    image_paths: Union[str, List[str]],
    context_text: str = "This is a room interior.",
) -> Union[List[float], List[List[float]]]:
    """
    Given one or more image paths, return Voyage embeddings.

    - If a single path (str) is passed, returns a single embedding list.
    - If a list of paths is passed, returns a list of embedding lists.
    """
    is_single = isinstance(image_paths, str)
    paths: List[str] = [image_paths] if is_single else image_paths

    embeddings = [embed_image_path(p, context_text) for p in paths]
    return embeddings[0] if is_single else embeddings




if __name__ == "__main__":
    image_path = "kitchen-calacatta-themis-01-v6.jpg"
    result = sam3_segment_image(image_path)
    saved_paths = process_segments_and_save(result, image_path)
    print(f"Saved {len(saved_paths)} segment images:")
    for path in saved_paths:
        print(f"  - {path}")
    embeddings = embed_segment_images(saved_paths)
    print(f"Generated {len(embeddings)} embeddings, each with {len(embeddings[0]) if embeddings else 0} dimensions")
    print(f"Embedding vector: {embeddings}")
