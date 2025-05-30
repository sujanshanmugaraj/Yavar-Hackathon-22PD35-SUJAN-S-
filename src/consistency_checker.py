from sentence_transformers import SentenceTransformer, util

class ConsistencyChecker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Load a sentence embedding model.
        """
        print(f"🔄 Loading sentence embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

    def check_similarity(self, caption1, caption2):
        """
        Compute cosine similarity between two captions.
        Returns a similarity score between 0 and 1.
        """
        emb1 = self.model.encode(caption1, convert_to_tensor=True)
        emb2 = self.model.encode(caption2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        return similarity

    def is_consistent(self, caption1, caption2, threshold=0.7):
        """
        Returns True if captions are consistent (similarity above threshold).
        """
        similarity = self.check_similarity(caption1, caption2)
        print(f"Similarity score: {similarity:.4f}")
        return similarity >= threshold


if __name__ == "__main__":
    checker = ConsistencyChecker()

    # Example captions
    concise_caption = "This chart shows temperature and rainfall over months."
    detailed_caption = "The chart displays monthly temperature and rainfall trends in Iowa county."

    consistent = checker.is_consistent(concise_caption, detailed_caption)
    if consistent:
        print("✅ Captions are consistent.")
    else:
        print("❌ Captions may be contradictory.")
