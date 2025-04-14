import numpy as np

from common.data import KlueDataLoader
from model.src.model import RoBertaModel


class Evaluator:
    def __init__(self):
        self.data_loader = KlueDataLoader(sample=False)
        self.model = RoBertaModel()

    def __repr__(self):
        return f"Evaluator({self.model.repo})"

    def run(self):
        _, test = self.data_loader.train_test()
        similarities = []
        labels = []

        for data in test:
            emb1 = self.model.embedding(data.sentence1)[0]
            emb2 = self.model.embedding(data.sentence2)[0]
            similarity = max(self.cosine_similarity(emb1, emb2), 0.0)
            label = data.real_label
            similarities.append(similarity)
            labels.append(label)

        print(f"\ncosine similarity:\t{similarities[: 5]}")
        print(f"labels:\t\t\t\t{labels[: 5]}")

        print(f"person correlation: {self.pearson_correlation(similarities, labels)}")

    @staticmethod
    def cosine_similarity(emb1: list[float], emb2: list[float]) -> float:
        """ 두 임베딩 사이의 cosine similarity 계산 """
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        cosine = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return cosine.item()

    @staticmethod
    def pearson_correlation(emb1: list[float], emb2: list[float]) -> float:
        """ 두 cosine similarity 사이의 pearson correlation 계산 """
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        correlation = np.corrcoef(emb1, emb2)[0, 1]
        return correlation.item()
