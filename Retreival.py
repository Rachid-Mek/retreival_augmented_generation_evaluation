from datasets import load_dataset
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Retrieval:
    def __init__(self, dataset_name, split):
        self.dataset = load_dataset(dataset_name, split=split)
        self.df = self.dataset.to_pandas()

    @staticmethod
    def compute_hit_rate(relevant_documents):
        """
        Calculate the Hit Rate for a retrieval augmented generation system.
        """
        return 1 if any(relevant_documents) else 0

    def hit_rate(self, row):
        row['hit_rate'] = self.compute_hit_rate(row['relvent_doc'])
        return row

    @staticmethod
    def reciprocal_rank(relevant_docs):
        """
        Calculate the Reciprocal Rank (MRR) for a retrieval augmented generation system.
        """
        if not relevant_docs or all(score == 0 for score in relevant_docs):
            return 0.0
        for i, score in enumerate(relevant_docs):
            if score == 1:
                return 1 / (i + 1)
        return 0.0

    def calculate_rr(self, row):
        row['mrr'] = self.reciprocal_rank(row['relvent_doc'])
        return row

    @staticmethod
    def ideal_dcg_at_k(relevances):
        sorted_relevances = sorted(relevances, reverse=True)
        ideal_dcg = sum([relevance / math.log2(i + 2) for i, relevance in enumerate(sorted_relevances)])
        return ideal_dcg

    @staticmethod
    def ndcg_at_k(relevant_docs, k):
        relevances = relevant_docs
        dcg = sum([relevance / math.log2(i + 2) for i, relevance in enumerate(relevances[:k])])
        idcg = Retrieval.ideal_dcg_at_k(relevances[:k])
        if idcg == 0:
            return 0
        return dcg / idcg

    def ndcg(self, row):
        row['ndcg'] = self.ndcg_at_k(row['relvent_doc'], row['num_chunks'])
        return row

    def preprocess_dataset(self):
        '''
        Preprocess the dataset by adding the hit_rate, mrr and ndcg columns for each row.
        '''
        self.dataset = self.dataset.map(self.hit_rate)
        self.dataset = self.dataset.map(self.calculate_rr)
        self.dataset = self.dataset.map(self.ndcg)

    def push_to_hub(self, hub_name):
        self.dataset.push_to_hub(hub_name)

    def visualize_metrics(self):
        '''
        Visualize the metrics calculated on the dataset on a bar plot.
        '''
        df = pd.DataFrame(self.dataset)
        print('Hit_rate:', round(df['hit_rate'].mean(), 3))
        print('MRR:', round(df['mrr'].mean(), 3))
        print('NDCG:', round(df['ndcg'].mean(), 3))

        plt.figure(figsize=(8, 6))
        sns.barplot(y=['Hit Rate', 'MRR', 'NDCG'], x=[df['hit_rate'].mean(), df['mrr'].mean(), df['ndcg'].mean()])
        plt.title("Le moyen obtenue pour chaque métrique d'évaluation")
        plt.xlabel('valeur')
        plt.show()

# Example usage:
retrieval = Retrieval('rachid16/Retrival_evaluation_dataset', 'train')
retrieval.preprocess_dataset()
# retrieval.push_to_hub("Retrival_evaluation_dataset_scores")
retrieval.visualize_metrics()