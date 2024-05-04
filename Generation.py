from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer


def calculate_bleu_score(machine_results, reference_texts):
  """
  Calculates BLEU score for machine translation outputs.

  Args:
      machine_results: A list of strings containing the machine-generated translations.
      reference_texts: A list of strings representing the human-written reference translations.

  Returns:
      None (prints the BLEU score).
  """

  # Split each reference text and machine translation result into individual words.
  reference_texts = [[ref.split() for ref in reference_texts]]  # List of lists for corpus_bleu
  machine_translations = [gen.split() for gen in machine_results]
  bleu_score = corpus_bleu(reference_texts, machine_translations)

  return bleu_score



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



def calculate_bert_score(generated_answers, ground_truth):
  """
  Calculates BERTScore for machine translation outputs.

  Args:
      generated_answers: A list of strings containing the machine-generated translations.
      ground_truth: A list of strings representing the human-written reference translations.

  Returns:
      None (prints BERTScore Precision, Recall, and F1 scores).
  """

  # Create a BERTScorer object with 'bert-base-uncased' pre-trained model for semantic similarity evaluation.
  scorer = BERTScorer(model_type='bert-base-uncased')
  # This line creates an instance of BERTScorer, likely utilizing a library.
  # The 'bert-base-uncased' model type specifies a pre-trained transformer model used for semantic similarity assessment.

  # Calculate Precision, Recall, and F1 score for each pair of generated answer and ground truth reference.
  P, R, F1 = scorer.score(generated_answers, ground_truth)
  avg_precision = sum(p.mean() for p in P) / len(P)
  avg_recall = sum(r.mean() for r in R) / len(R)
  avg_f1 = sum(f1.mean() for f1 in F1) / len(F1)

  return avg_precision, avg_recall, avg_f1