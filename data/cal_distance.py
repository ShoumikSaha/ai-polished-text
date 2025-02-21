import textdistance
import torch
from transformers import AutoTokenizer, AutoModel


class SimCalculator:
    def __init__(self, model_name="sci_bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def bert_similarity(self, text1, text2):
        inputs1 = self.tokenizer(text1, return_tensors="pt")
        inputs2 = self.tokenizer(text2, return_tensors="pt")
        inputs1 = inputs1.to(self.device)
        inputs2 = inputs2.to(self.device)        
        with torch.no_grad():
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)

        embeddings1 = outputs1.last_hidden_state[:, 0, :]
        embeddings2 = outputs2.last_hidden_state[:, 0, :]
        cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).cpu()
        return cos_sim.item()

    def cal_similarity(self, text1, text2):
        split_1 = text1.split(" ")
        split_2 = text2.split(" ")
        try:
            levenshtein_distance = textdistance.levenshtein.normalized_distance(split_1, split_2)
            jaccard_distance = textdistance.jaccard.normalized_distance(split_1, split_2)
            sem_similarity = self.bert_similarity(text1, text2)
            return_dict = {
                "levenshtein_distance": round(levenshtein_distance, 4),
                "jaccard_distance": round(jaccard_distance, 4),
                "sem_similarity": round(sem_similarity, 4)
            }
        except:
            return_dict = {
                "levenshtein_distance": 1.0,
                "jaccard_distance": 1.0,
                "sem_similarity": 0.0
            }
        return return_dict