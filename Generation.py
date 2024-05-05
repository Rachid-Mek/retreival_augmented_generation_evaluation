from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import nltk
nltk.download('punkt')


class Generation:
    def __init__(self):
        # Initialize the BERTScorer only once, assuming it's expensive to load.
        self.bert_scorer = BERTScorer(model_type='bert-base-uncased')
    
    def calculate_bleu_score(self, machine_results, reference_texts):
        """
        Calculates BLEU score for machine translation outputs.
        """
        tokenized_references = [[nltk.word_tokenize(ref)] for ref in reference_texts]
        tokenized_machines = [nltk.word_tokenize(result) for result in machine_results]
        return corpus_bleu(tokenized_references, tokenized_machines)
    
    def calculate_rouge_scores(self, generated_answers, ground_truth):
        """
        Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for machine translation outputs.
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        total_rouge1, total_rouge2, total_rougeL = 0, 0, 0

        for gen, ref in zip(generated_answers, ground_truth):
            scores = scorer.score(gen, ref)
            total_rouge1 += scores['rouge1'].fmeasure
            total_rouge2 += scores['rouge2'].fmeasure
            total_rougeL += scores['rougeL'].fmeasure

        num_answers = len(generated_answers)
        return (total_rouge1 / num_answers, total_rouge2 / num_answers, total_rougeL / num_answers)
    
    def calculate_bert_score(self, dataset, generated_answers, ground_truth):
        """
        Calculates BERT scores for machine translation outputs and appends results to the dataset.
        """
        P, R, F1 = self.bert_scorer.score(generated_answers, ground_truth)
        bertscore_precision = [round(p.mean().item(), 4) for p in P]
        bertscore_recall = [round(r.mean().item(), 4) for r in R]
        bertscore_f1 = [round(f1.mean().item(), 4) for f1 in F1]

        dataset['BERTScore_Precision'] = bertscore_precision
        dataset['BERTScore_Recall'] = bertscore_recall
        dataset['BERTScore_F1'] = bertscore_f1

        avg_precision = sum(bertscore_precision) / len(bertscore_precision)
        avg_recall = sum(bertscore_recall) / len(bertscore_recall)
        avg_f1 = sum(bertscore_f1) / len(bertscore_f1)

        return dataset, avg_precision, avg_recall, avg_f1