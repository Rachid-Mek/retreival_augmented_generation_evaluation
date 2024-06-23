
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from datasets import Dataset , load_dataset
import pandas as pd
import nltk 
import torch
nltk.download('punkt')
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
# ====================================================================================================

def calculate_bleu_score(machine_results, reference_texts):
    """
    Calculates BLEU score for machine translation outputs.

    Args:
        machine_results: A list of strings containing the machine-generated translations.
        reference_texts: A list of strings representing the human-written reference translations.

    Returns:
        BLEU score as a float.
    """
    # Ensure each reference is a list of words (tokenize if necessary)
    tokenized_references = [[nltk.word_tokenize(ref)] for ref in reference_texts]  # List of lists of lists
    tokenized_machines = [nltk.word_tokenize(result) for result in machine_results]  # List of lists

    # Calculate BLEU score
    bleu_score = corpus_bleu(tokenized_references, tokenized_machines)
    return bleu_score

def bleu_score(row):
    if "FT_response" in row:
      machine_results = row["FT_response"]
    else:
      machine_results = row["NFT_response"]
    reference_texts = row["answer"]
    bleu = calculate_bleu_score(machine_results, reference_texts)
    row["bleu_score"] = bleu
    return row

def bleu_score_dataset(dataset):
    dataset = dataset.map(bleu_score)
    return dataset

# ====================================================================================================

def calculate_rouge_scores(generated_answers, ground_truth):
  """
  Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for machine translation outputs.

  Args:
      generated_answers: A list of strings containing the machine-generated translations.
      ground_truth: A list of strings representing the human-written reference translations.

  Returns:
      None (prints the average ROUGE-1, ROUGE-2, and ROUGE-L scores).
  """

  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
  total_rouge1, total_rouge2, total_rougeL = 0, 0, 0

  # Iterate through each pair of generated answer and its corresponding ground truth reference.
  for gen, ref in zip(generated_answers, ground_truth):
    # Calculate individual ROUGE-1, ROUGE-2, and ROUGE-L scores for each pair using the scorer object.
    scores = scorer.score(gen, ref)
    total_rouge1 += scores['rouge1'].fmeasure
    total_rouge2 += scores['rouge2'].fmeasure
    total_rougeL += scores['rougeL'].fmeasure

  # Calculate average scores for each ROUGE metric by dividing the total scores by the number of generated answers.
  average_rouge1 = total_rouge1 / len(generated_answers)
  average_rouge2 = total_rouge2 / len(generated_answers)
  average_rougeL = total_rougeL / len(generated_answers)
  return average_rouge1, average_rouge2, average_rougeL

def rouge_score(row):
  generated_answers = row["Finetuned_answer"]
  ground_truth = row["answer"]
  average_rouge1, average_rouge2, average_rougeL = calculate_rouge_scores(generated_answers, ground_truth)
  row["rouge1_score"] = average_rouge1
  row["rouge2_score"] = average_rouge2
  row["rougeL_score"] = average_rougeL
  return row

def rouge_score_dataset(dataset):
    dataset = dataset.apply(rouge_score, axis=1)
    return dataset
# ====================================================================================================


scorer = BERTScorer(model_type='bert-base-uncased')


def calculate_bert_score(dataset, generated_answers, ground_truth):
    P, R, F1 = scorer.score(generated_answers, ground_truth)

    bertscore_precision = []
    bertscore_recall = []
    bertscore_f1 = []

    for (p, r, f1) in zip(P, R, F1):
        bertscore_precision.append(round(p.mean().item(), 4))
        bertscore_recall.append(round(r.mean().item(), 4))
        bertscore_f1.append(round(f1.mean().item(), 4))

    dataset['BERTScore_Precision'] = bertscore_precision
    dataset['BERTScore_Recall'] = bertscore_recall
    dataset['BERTScore_F1'] = bertscore_f1
    avg_precision = sum(bertscore_precision) / len(bertscore_precision)
    avg_recall = sum(bertscore_recall) / len(bertscore_recall)
    avg_f1 = sum(bertscore_f1) / len(bertscore_f1)

    return dataset, avg_precision, avg_recall, avg_f1

def bert_score(row):
  ground_truth = row["answer"]
  generated_text = row["Finetuned_answer"]
  P, R, F1 = scorer.score([generated_text], [ground_truth])
  row["bert_score"] = F1.mean().item()
  return row

def bert_score_dataset(dataset):
    dataset = dataset.apply(bert_score, axis=1)
    return dataset
# ====================================================================================================


model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_entailment(premise, hypothesis):

  """Predicts the probability of entailment between two sentences using DeBERTa.

  Args :
  premise: The first sentence (premise).
  hypothesis: The second sentence (hypothesis).

  Returns:
  The probability of the 'entailment' label."""

  # Preprocess sentences (tokenization)
  inputs = tokenizer(premise, hypothesis, return_tensors="pt")

  # Perform prediction
  outputs = model( ** inputs)
  predictions = F.softmax(outputs.logits, dim =- 1) # Get probabilities for each label
  entailment_prob = predictions[0, 0].item() # Index 0 for 'entailment'

  return entailment_prob

def entailement(row):
  premise = row['context']
  hypothesis = row['Finetuned_answer']
  entailment_prob = predict_entailment(premise, hypothesis)
  print(entailment_prob)
  print('-----------------------------------------')
  row["entailement_score"] = entailment_prob
  return row

def entailement_dataset(dataset):
    dataset = dataset.apply(entailement, axis=1)
    return dataset

# ====================================================================================================


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")

def get_bert_embeddings(sentence, tokenizer, model, use_sentence_transformer=False):
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

def compute_bert_similarity(sentence1, sentence2, tokenizer, model, use_sentence_transformer=False):
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
      sentence1_embedding = get_bert_embeddings(sentence1, tokenizer, model, use_sentence_transformer)
      sentence2_embedding = get_bert_embeddings(sentence2, tokenizer, model, use_sentence_transformer)
      sentence_embeddings = [sentence1_embedding, sentence2_embedding]

  # Calculate cosine similarity between sentence embeddings
  cosine_similarity = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item()

  return cosine_similarity

def bert_score(row):
  ground_truth = row["answer"]
  generated_text = row["Finetuned_answer"]
  bert_similarity = compute_bert_similarity(ground_truth, generated_text, tokenizer, model)
  row["bert_similarity_score"] = bert_similarity
  return row

def bert_score_dataset(dataset):
    dataset = dataset.apply(bert_score, axis=1)
    return dataset

# ====================================================================================================
 
