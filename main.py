from Generation import Generation
from utils import extract_text
from datasets import load_dataset , Dataset
import pandas as pd

# Load the data
dataset = load_dataset('rachid16/ft_bert_benchmark', split ='train')
dataset_df = pd.DataFrame(dataset)
dataset_df.head()

# Extract the reference texts and machine-generated answers
reference_texts=list(dataset_df["answer"])
machine_results_Finetuned= list(dataset_df['Finetuned_answer'])

# preprocessing
machine_results_Finetuned_Copy = extract_text(machine_results_Finetuned)
dataset_copy = dataset_df.copy()

if __name__ == '__main__':
    # instantiate the Generation class
    evaluation = Generation()
    # dataset_copy , p , r , f1 = evaluation.calculate_bert_score(dataset_copy ,machine_results_Finetuned_Copy, reference_texts)
    # print("Average Precision:", p)
    # print("Average Recall:", r)
    # print("Average F1 Score:", f1)

    # dataset_copy.to_csv('ft_bert_benchmark.csv')
    # dataset_copy = Dataset.from_pandas(dataset_copy)
    # dataset_copy.push_to_hub('ft_bert_benchmark')

    dataset_copy = dataset.map(evaluation.entailement)
    dataset_copy.push_to_hub('ft_bert_benchmark')