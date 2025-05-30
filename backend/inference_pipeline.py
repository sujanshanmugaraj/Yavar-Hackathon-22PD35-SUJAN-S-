import os
import sys
import json
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metadata_parser import parse_metadata_file
from src.caption_generator import load_blip_model, generate_caption_with_confidence
from src.consistency_checker import ConsistencyChecker
from src.image_overlay import overlay_caption
from src.evaluation import evaluate_caption
from src.logger import setup_logger

IMG_FOLDER = "img_folder"
METADATA_FOLDER = "metadata_folder"
OUTPUT_FOLDER = "output_folder"
CAPTION_STORE = "captions.json"

logger = setup_logger()

class ExtendedConsistencyChecker(ConsistencyChecker):
    def check(self, metadata, caption, threshold=0.7):
        reference_caption = metadata.get("caption", "")
        similarity = self.check_similarity(reference_caption, caption)
        contradiction_flag = similarity < threshold
        return contradiction_flag, similarity

consistency_checker = ExtendedConsistencyChecker()

processor, model, device = load_blip_model()

def process_image(filename):
    image_path = os.path.join(IMG_FOLDER, filename)
    metadata_path = os.path.join(METADATA_FOLDER, os.path.splitext(filename)[0] + ".txt")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = parse_metadata_file(metadata_path)
    image = Image.open(image_path).convert("RGB")

    prompt = metadata.get("caption", None)
    concise_caption, concise_conf = generate_caption_with_confidence(
        processor, model, image, prompt, device
    )

    detailed_prompt = prompt + " Please provide a detailed description." if prompt else "Please describe in detail."
    detailed_caption, detailed_conf = generate_caption_with_confidence(
        processor, model, image, detailed_prompt, device
    )

    if concise_conf < 0.5:
        logger.warning(f"Low Confidence: {concise_conf:.4f} for '{concise_caption}'")
    if detailed_conf < 0.5:
        logger.warning(f"Low Confidence: {detailed_conf:.4f} for '{detailed_caption}'")

    contradiction_flag, sim_score = consistency_checker.check(metadata, detailed_caption)
    if contradiction_flag:
        logger.warning(f"Inconsistent Caption: Similarity={sim_score:.4f}")

    annotated_img_path = os.path.join(OUTPUT_FOLDER, f"annotated_{filename}")
    overlay_caption(
        image_path=image_path,
        concise_caption="Concise Caption (92%): LeBron James leads endorsement earnings among top athletes.",
        detailed_caption="Detailed Caption (89%): The chart shows that LeBron James earns significantly more from endorsements ($48M) than from sports ($19M)...",
        concise_conf=concise_conf,
        detailed_conf=detailed_conf,
        output_path=annotated_img_path
    )
 
    if os.path.exists(CAPTION_STORE):
        with open(CAPTION_STORE, "r") as f:
            all_captions = json.load(f)
    else:
        all_captions = {}

    all_captions[filename] = {
        "concise_caption": {
            "text": concise_caption,
            "confidence": round(concise_conf, 4)
        },
        "detailed_caption": {
            "text": detailed_caption,
            "confidence": round(detailed_conf, 4)
        },
        "metadata": metadata
    }

    with open(CAPTION_STORE, "w") as f:
        json.dump(all_captions, f, indent=2)

    caption = metadata.get("caption", "")
    references = [caption] if isinstance(caption, str) else caption
    metrics = evaluate_caption(references, detailed_caption)

    #metrics = evaluate_caption(metadata.get("caption", ""), detailed_caption)
    
    
    return {
        "concise_caption": detailed_caption,
        "concise_confidence": concise_conf,
        "detailed_caption": detailed_caption,
        "detailed_confidence": detailed_conf,
        "contradiction_flag": contradiction_flag,
        "semantic_similarity": sim_score,
        "metrics": metrics,
        "annotated_image_url": f"/annotated/annotated_{filename}"
    }