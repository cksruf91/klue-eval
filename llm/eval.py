from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common.data import KlueDataLoader
from llm.client import OpenAIClient


class Evaluator:
    def __init__(self):
        self.data_loader = KlueDataLoader(sample=False)
        self.client = OpenAIClient(model='text-embedding-3-small')

    def run(self):
        _, test = self.data_loader.train_test()
        similarities = []
        labels = []

        print("get similarity...")
        with ThreadPoolExecutor(10) as executor:
            results = list(executor.map(self.process_data, test))
            for similarity, label in results:
                similarities.append(similarity)
                labels.append(label)

        print(f"\ncosine similarity:\t{similarities[: 5]}")
        print(f"labels:\t\t\t\t{labels[: 5]}")

        print(f"person correlation: {self.pearson_correlation(similarities, labels)}")
        # 0.735812685645248, 0.7354876802957254, 0.7355130002574058

    def process_data(self, data):
        emb1 = self.client.embedding(text=data.sentence1)
        emb2 = self.client.embedding(text=data.sentence2)
        similarity = max(self.cosine_similarity(emb1, emb2), 0.0)
        return similarity, data.real_label

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
