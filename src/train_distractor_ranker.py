import os
import random
import numpy as np
import pandas as pd
import nltk
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def _char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower()), set(b.lower())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0

def _word_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0

def compute_features(candidate: str, correct_answer: str, article: str,
                     ohe_vec, tfidf_vec) -> list:

    try:
        ohe_sim = cosine_similarity(
            ohe_vec.transform([candidate]),
            ohe_vec.transform([correct_answer])
        )[0][0]

        char_match = _char_jaccard(candidate, correct_answer)
        freq = article.lower().count(candidate.lower())

        tfidf_sim = cosine_similarity(
            tfidf_vec.transform([candidate]),
            tfidf_vec.transform([correct_answer])
        )[0][0]

        word_match = _word_jaccard(candidate, correct_answer)
        len_ratio = min(len(candidate.split()) / max(len(correct_answer.split()), 1), 5.0) / 5.0

        return [ohe_sim, char_match, freq, tfidf_sim, word_match, len_ratio]

    except Exception:
        return [0.0, 0.0, 0, 0.0, 0.0, 0.0]

def train_ranker(input_path: str = 'data/raw/train.csv',
                 nrows: int = 500,
                 neg_per_row: int = 4,
                 random_seed: int = 42) -> None:

    random.seed(random_seed)
    np.random.seed(random_seed)

    print("=" * 55)
    print("  Training Distractor Ranker")
    print("=" * 55)

    if not os.path.exists(input_path):
        print(f"[ERROR] Dataset not found at '{input_path}'. Aborting.")
        return

    df = pd.read_csv(input_path, nrows=nrows)
    print(f"  Loaded {len(df)} rows from '{input_path}'")

    all_text = (
        df['article'].fillna("").tolist()
        + df['question'].fillna("").tolist()
        + df['A'].fillna("").tolist()
        + df['B'].fillna("").tolist()
        + df['C'].fillna("").tolist()
        + df['D'].fillna("").tolist()
    )

    print("  Fitting vectorizers...")
    ohe_vec = CountVectorizer(binary=True, max_features=8000)
    ohe_vec.fit(all_text)

    tfidf_vec = TfidfVectorizer(max_features=8000)
    tfidf_vec.fit(all_text)

    X, y = [], []
    options = ['A', 'B', 'C', 'D']

    for _, row in df.iterrows():
        article     = str(row.get('article', ''))
        answer_col  = str(row.get('answer', '')).strip()
        correct_ans = str(row.get(answer_col, ''))

        sentences = nltk.sent_tokenize(article)
        option_texts = {opt: str(row.get(opt, '')) for opt in options}

        for opt in options:
            if opt == answer_col:
                continue
            distractor = option_texts[opt]
            if len(distractor.split()) < 2:
                continue
            feat = compute_features(distractor, correct_ans, article, ohe_vec, tfidf_vec)
            X.append(feat)
            y.append(1)

        option_set = set(option_texts.values())
        candidate_negs = [
            s for s in sentences
            if len(s.split()) > 3 and s not in option_set
        ]

        sampled_negs = random.sample(candidate_negs, min(neg_per_row, len(candidate_negs)))
        for sent in sampled_negs:
            feat = compute_features(sent, correct_ans, article, ohe_vec, tfidf_vec)
            X.append(feat)
            y.append(0)

    X = np.array(X)
    y = np.array(y)

    pos = y.sum()
    print(f"  Samples: positives={int(pos)}, negatives={len(y) - int(pos)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_seed, stratify=y
    )

    print("  Training LogisticRegression (class_weight=balanced)...")
    model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=random_seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n  Evaluation on held-out 15%:")
    print(classification_report(y_test, y_pred, target_names=['Not distractor', 'Good distractor']))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model,    'models/distractor_ranker.pkl')
    joblib.dump(ohe_vec,  'models/distractor_ohe.pkl')
    joblib.dump(tfidf_vec,'models/distractor_tfidf.pkl')

    joblib.dump(tfidf_vec,'models/tfidf_vectorizer_checkpoint.pkl')

    print("\n  Saved:")
    print("    models/distractor_ranker.pkl")
    print("    models/distractor_ohe.pkl")
    print("    models/distractor_tfidf.pkl")
    print("    models/tfidf_vectorizer_checkpoint.pkl (fallback for gen file)")
    print("=" * 55)
    print("  Done!")

if __name__ == "__main__":
    train_ranker()
