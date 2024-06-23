from pkgutil import get_data
from altair import Datasets
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
from datasets import Dataset
import re
sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")
nltk.download('punkt')


class Generation_Evaluation():
    def __init__(self):
        self.bert_scorer = BERTScorer(model_type='bert-base-uncased')
        self.tokenizer_ = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-MNLI')
        self.model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-MNLI')
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.tokenizer_deberta = AutoTokenizer.from_pretrained(model_name)
        self.model_deberta = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval() 
        if torch.cuda.is_available():
            self.model.cuda()  # Transfer the model to GPU if available

    def load_data_ft(self):
        self.dataset = load_dataset('islam23/ft_eval_data' , split='train')
        return self.dataset
    def load_data_nft(self):
        self.dataset = load_dataset('islam23/nft_eval_data' , split='train')
        return self.dataset


    def extract_text(self ,results):
    
        pattern = r"(According to|Based on)\s+.*?,"  # Matches "According to" or "Based on" followed by anything until a comma (,)
        extracted_text = []
        for result in results:
            # Substitute the matched pattern with an empty string
            cleaned_text = re.sub(pattern, "", result, flags=re.IGNORECASE)
            extracted_text.append(cleaned_text.strip())  # Remove leading/trailing whitespaces

        return extracted_text
    
    staticmethod
    def bleu_score( row):
        if "FT_response" in row:
            generated_result = row["FT_response"]
        else:
            generated_result = row["NFT_response"]
       
        reference_texts = row["answer"]

        bleu_score = corpus_bleu([[ref.split()] for ref in reference_texts], [gen.split() for gen in generated_result])

        row['Bleu'] = bleu_score
        return row
    
    def bleu_score_dataset(self):
        dataset_df = pd.DataFrame(self.dataset)
        if "FT_response" in dataset_df:
            machine_results= list(dataset_df['FT_response'])
        else:
            machine_results= list(dataset_df['NFT_response'])
        machine_results_Finetuned_Copy = self.extract_text(machine_results)
        if "FT_response" in dataset_df:
            dataset_df["FT_response"]=machine_results_Finetuned_Copy
        else:
            dataset_df["NFT_response"]=machine_results_Finetuned_Copy
        dataset = Dataset.from_pandas(dataset_df)
        dataset = self.dataset.map(Generation_Evaluation.bleu_score)
        return dataset
    

    def get_data(self):
        return self.dataset

#====================================================================================================
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
    
    def rouge_score(self, row):
        if "FT_response" in row:
            generated_answers = row["FT_response"]
        else:
            generated_answers = row["NFT_response"]
        ground_truth = row["answer"]
        average_rouge1, average_rouge2, average_rougeL = self.calculate_rouge_scores(generated_answers, ground_truth)
        row["rouge1_score"] = average_rouge1
        row["rouge2_score"] = average_rouge2
        row["rougeL_score"] = average_rougeL
        return row
    
    def rouge_score_dataset(self):
        dataset = self.dataset.map(self.rouge_score)
        return dataset
    
#====================================================================================================
    
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
    

    
    def bert_score_dataset(self):
        dataset_df = pd.DataFrame(self.dataset)
      
        reference_texts=list(dataset_df["answer"])
        if "FT_response" in dataset_df:
            machine_results_Finetuned= list(dataset_df['FT_response'])
        else:
            machine_results_Finetuned= list(dataset_df['NFT_response'])
        dataset_df,avg_precision, avg_recall, avg_f1 = self.calculate_bert_score(dataset_df, machine_results_Finetuned, reference_texts)
 
 
        # self.dataset = Dataset.to_dict(dataset_df)
        return self.dataset, avg_precision, avg_recall, avg_f1
    
#====================================================================================================


    def predict_entailment(self, premise, hypothesis):
        """
        Predicts the probability of entailment between two sentences using DeBERTa.

        Args:
            premise: The first sentence (premise).
            hypothesis: The second sentence (hypothesis).

        Returns:
            The probability of the 'entailment' label.
        """
 
        inputs = self.tokenizer_deberta(premise, hypothesis, return_tensors="pt")

        # Perform prediction
        outputs = self.model_deberta( ** inputs)
        predictions = F.softmax(outputs.logits, dim =- 1) # Get probabilities for each label
        entailment_prob = predictions[0, 0].item() # Index 0 for 'entailment'


        return entailment_prob
    
    def entailement(self, row):
        premise = row["context"]
        if "FT_response" in row:
            hypothesis = row["FT_response"]
        else:
            hypothesis = row["NFT_response"]
        entailment_prob = self.predict_entailment(premise, hypothesis)
        row["entailment_prob"] = entailment_prob
        return row
    
    def entailement_dataset(self):
        dataset = self.dataset.map(self.entailement)
        return dataset
    
#====================================================================================================


    def get_bert_embeddings(self , sentence, tokenizer, model, use_sentence_transformer=False):
        """
        Computes the BERT embeddings for a given sentence.

        Args:
            sentence: The input sentence.
            tokenizer: The pre-trained BERT tokenizer.
            model: The pre-trained BERT model.
            use_sentence_transformer: Whether to use SentenceTransformer for encoding.

        Returns:
            The BERT embeddings for the input sentence.
        """
        # Encode sentence
        if use_sentence_transformer:
            sentence_embedding = sentence_transformer_model.encode(sentence, convert_to_tensor=True)
        else:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                sentence_embedding = outputs.last_hidden_state[:, 0, :]

        return sentence_embedding

    def compute_bert_similarity(self , sentence1, sentence2, tokenizer, model, use_sentence_transformer=False):
        """
        Computes the cosine similarity between sentence embeddings using BERT.

        Args:
            sentence1: The first sentence.
            sentence2: The second sentence.
            tokenizer: The pre-trained BERT tokenizer.
            model: The pre-trained BERT model.
            use_sentence_transformer: Whether to use SentenceTransformer for encoding.

        Returns:
            The cosine similarity between the embeddings of the two sentences.
        """
        # Encode sentences
        if use_sentence_transformer:
            sentence_embeddings = sentence_transformer_model.encode([sentence1, sentence2], convert_to_tensor=True)
        else:
            with torch.no_grad():
            # Get embeddings from separate function for clarity
                sentence1_embedding = self.get_bert_embeddings(sentence1, tokenizer, model, use_sentence_transformer)
                sentence2_embedding = self.get_bert_embeddings(sentence2, tokenizer, model, use_sentence_transformer)
            sentence_embeddings = [sentence1_embedding, sentence2_embedding]

        # Calculate cosine similarity between sentence embeddings
        cosine_similarity = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item()

        return cosine_similarity

    def bert_score(self , row):
        ground_truth = row["answer"]
        if "FT_response" in row:
            generated_text = row["FT_response"]
        else:
            generated_text = row["NFT_response"]
        bert_similarity = self.compute_bert_similarity(ground_truth, generated_text, tokenizer, model)
        row["bert_similarity_score"] = bert_similarity
        return row
    
    def bert_dataset(self):
        dataset = self.dataset.map(self.bert_score)
        return dataset
    
#====================================================================================================