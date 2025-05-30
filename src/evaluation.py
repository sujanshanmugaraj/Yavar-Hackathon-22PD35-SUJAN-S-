
import json
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

_model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_caption(references, generated_caption):
    smoothie = SmoothingFunction().method4

    bleu = sentence_bleu(
        [ref.split() for ref in references],
        generated_caption.split(),
        smoothing_function=smoothie
    )

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(' '.join(references), generated_caption)

    ref_embedding = _model.encode(' '.join(references), convert_to_tensor=True)
    gen_embedding = _model.encode(generated_caption, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(ref_embedding, gen_embedding).item()

    return {
        "BLEU": round(bleu, 4),
        "ROUGE_1": round(rouge['rouge1'].fmeasure, 4),
        "ROUGE_L": round(rouge['rougeL'].fmeasure, 4),
        "Semantic_Similarity": round(cosine_sim, 4)
    }

def load_references(reference_dir):
    ref_data = {}
    for file in os.listdir(reference_dir):
        if file.endswith(".txt"):
            img_id = os.path.splitext(file)[0]
            with open(os.path.join(reference_dir, file), "r") as f:
                refs = f.read().strip().split('\n')
                ref_data[img_id] = [r.strip() for r in refs if r.strip()]
    return ref_data

if __name__ == "__main__":
    with open("captions.json") as f:
        captions = json.load(f)

    reference_dir = "metadata_folder"  
    references = load_references(reference_dir)

    for img_id, cap_data in captions.items():
        ref_texts = references.get(img_id, [])
        if not ref_texts or not cap_data.get("detailed_caption"):
            continue

        # generated = cap_data["detailed_caption"]
        # scores = evaluate_caption(ref_texts, generated)
        detailed_data = cap_data.get("detailed_caption", {})
        generated = detailed_data.get("text", "").strip()

        if not generated or not ref_texts:
            continue

        scores = evaluate_caption(ref_texts, generated)

        cap_data["evaluation_metrics"] = scores
        cap_data["contradiction"] = scores["Semantic_Similarity"] < 0.5


        captions[img_id].update(scores)

        captions[img_id]["Contradiction"] = scores["Semantic_Similarity"] < 0.5

    with open("captions.json", "w") as f:
        json.dump(captions, f, indent=2)

    print("✅ Evaluation metrics added to captions.json")
