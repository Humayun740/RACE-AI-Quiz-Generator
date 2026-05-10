import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

def evaluate_generation(n_samples: int = 100):
    raw_path = 'data/raw/train.csv'
    if not os.path.exists(raw_path):
        print(f"[ERROR] {raw_path} not found.")
        return {}

    df = pd.read_csv(raw_path, nrows=n_samples)

    from distraction_gen import generate_distractors

    scorer   = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method1

    bleu_scores, rouge_scores, meteor_scores = [], [], []
    skipped = 0

    print(f"Evaluating {len(df)} samples...")
    print("=" * 50)

    for _, row in df.iterrows():
        try:
            correct_label  = row['answer']
            all_options    = ['A', 'B', 'C', 'D']
            distractor_labels = [opt for opt in all_options if opt != correct_label]

            gold_distractors = [str(row[lbl]) for lbl in distractor_labels]
            correct_answer   = str(row[correct_label])
            article          = str(row['article'])

            generated = generate_distractors(article, correct_answer, num_distractors=3)

            if not generated:
                skipped += 1
                continue

            sample_bleu, sample_rouge, sample_meteor = [], [], []

            for gen in generated:
                gen_tokens  = gen.lower().split()
                ref_tokens  = [g.lower().split() for g in gold_distractors]

                bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothie)
                sample_bleu.append(bleu)

                r = max(scorer.score(gold, gen)['rougeL'].fmeasure for gold in gold_distractors)
                sample_rouge.append(r)

                meteor_scores_per_ref = [
                    meteor_score([g.split()], gen_tokens) for g in gold_distractors
                ]
                m = max(meteor_scores_per_ref) if meteor_scores_per_ref else 0.0
                sample_meteor.append(m)

            bleu_scores.append(sum(sample_bleu) / len(sample_bleu))
            rouge_scores.append(sum(sample_rouge) / len(sample_rouge))
            meteor_scores.append(sum(sample_meteor) / len(sample_meteor))

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                import traceback
                traceback.print_exc()

    total = len(bleu_scores)
    avg_bleu   = sum(bleu_scores)   / total if total else 0
    avg_rouge  = sum(rouge_scores)  / total if total else 0
    avg_meteor = sum(meteor_scores) / total if total else 0

    print("\nFINAL NLP GENERATION METRICS (Model B)")
    print("=" * 50)
    print(f"  Samples evaluated : {total} (skipped: {skipped})")
    print(f"  Average BLEU      : {avg_bleu:.4f}")
    print(f"  Average ROUGE-L   : {avg_rouge:.4f}")
    print(f"  Average METEOR    : {avg_meteor:.4f}")
    print("=" * 50)

    return {
        'bleu':    round(avg_bleu, 4),
        'rouge_l': round(avg_rouge, 4),
        'meteor':  round(avg_meteor, 4),
        'n':       total,
    }

if __name__ == "__main__":
    evaluate_generation(n_samples=100)
