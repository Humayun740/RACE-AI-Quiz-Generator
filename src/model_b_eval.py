import pandas as pd
import joblib
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import os

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

def evaluate_generation():
    test_path = 'data/processed/sample_test.csv'
    if not os.path.exists(test_path):
        df = pd.read_csv('data/raw/train.csv', nrows=50)
    else:
        df = pd.read_csv(test_path)

    from distraction_gen import generate_distractors
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method1
    
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    
    print(f"Evaluating {len(df.head(50))} samples for NLP Metrics...")
    
    for _, row in df.head(50).iterrows():
        correct_label = row['answer']
        all_options = ['A', 'B', 'C', 'D']
        distractor_labels = [opt for opt in all_options if opt != correct_label]
        
        references = [str(row[lbl]) for lbl in distractor_labels]
        
        generated_list = generate_distractors(row['article'], str(row[correct_label]))
        
        candidate = " ".join(generated_list) if isinstance(generated_list, list) and len(generated_list) > 0 else ""
        
        if not candidate:
            continue

        ref_tokens = [ref.split() for ref in references]
        can_tokens = candidate.split()
        bleu = sentence_bleu(ref_tokens, can_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu + 1e-1)

        r_scores = [scorer.score(ref, candidate)['rougeL'].fmeasure for ref in references]
        rouge_scores.append(max(r_scores))
        
        m_score = meteor_score([ref.split() for ref in references], can_tokens)
        meteor_scores.append(m_score)

    print("\n" + "="*30)
    print("FINAL NLP GENERATION METRICS")
    print("="*30)
    print(f"Average BLEU:   {sum(bleu_scores)/len(bleu_scores) if bleu_scores else 0:.4f}")
    print(f"Average ROUGE-L: {sum(rouge_scores)/len(rouge_scores) if rouge_scores else 0:.4f}")
    print(f"Average METEOR:  {sum(meteor_scores)/len(meteor_scores) if meteor_scores else 0:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_generation()