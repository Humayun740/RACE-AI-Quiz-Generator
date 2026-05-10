import os
import numpy as np
import pandas as pd
import nltk
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def _keyword_overlap(sentence: str, query: str) -> float:

    stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on',
            'at', 'to', 'for', 'of', 'and', 'or', 'but', 'it', 'that'}
    s_words = {w for w in sentence.lower().split() if w not in stop and len(w) > 2}
    q_words = {w for w in query.lower().split()    if w not in stop and len(w) > 2}
    if not s_words or not q_words:
        return 0.0
    return len(s_words & q_words) / len(s_words | q_words)

def _ohe_sim(sentence: str, query: str, vectorizer) -> float:

    try:
        vecs = vectorizer.transform([sentence, query])
        return float(cosine_similarity(vecs[0], vecs[1])[0][0])
    except Exception:
        return 0.0

def extract_features(sentence: str, query: str, position: int,
                     total: int, answer: str, vectorizer) -> list:

    kw_overlap    = _keyword_overlap(sentence, query)
    sent_position = position / max(total - 1, 1)           
    sent_length   = min(len(sentence.split()), 50) / 50.0  
    ohe_sim       = _ohe_sim(sentence, query, vectorizer)
    ans_words     = set(answer.lower().split())
    sent_words    = set(sentence.lower().split())
    ans_contains  = float(bool(ans_words & sent_words))

    return [kw_overlap, sent_position, sent_length, ohe_sim, ans_contains]

def train_hint_scorer(input_path: str = 'data/raw/train.csv',
                      nrows: int = 2000,
                      random_seed: int = 42) -> None:

    print("=" * 55)
    print("  Training Hint Scorer (Logistic Regression)")
    print("=" * 55)

    if not os.path.exists(input_path):
        print(f"[ERROR] Dataset not found at {input_path}")
        return

    df = pd.read_csv(input_path, nrows=nrows)
    print(f"  Loaded {len(df)} rows")

    all_text = (
        df['article'].fillna("").tolist()
        + df['question'].fillna("").tolist()
        + df['A'].fillna("").tolist()
        + df['B'].fillna("").tolist()
        + df['C'].fillna("").tolist()
        + df['D'].fillna("").tolist()
    )
    print("  Fitting OHE vectorizer...")
    ohe_vec = CountVectorizer(binary=True, max_features=10000)
    ohe_vec.fit(all_text)

    X, y = [], []
    skipped = 0

    for _, row in df.iterrows():
        article      = str(row.get('article', ''))
        question     = str(row.get('question', ''))
        answer_col   = str(row.get('answer', '')).strip()
        correct_ans  = str(row.get(answer_col, ''))
        query        = f"{question} {correct_ans}"

        sentences = nltk.sent_tokenize(article)
        sentences = [s.strip() for s in sentences if len(s.split()) > 4]
        if len(sentences) < 2:
            skipped += 1
            continue

        total = len(sentences)
        ans_lower = correct_ans.lower()

        for pos, sent in enumerate(sentences):

            label = 1 if ans_lower in sent.lower() else 0
            feats = extract_features(sent, query, pos, total, correct_ans, ohe_vec)
            X.append(feats)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    pos_count = y.sum()
    print(f"  Samples: positives={int(pos_count)}, negatives={len(y) - int(pos_count)}")
    print(f"  Skipped: {skipped} rows (too short)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_seed, stratify=y
    )

    print("  Training LogisticRegression...")
    model = LogisticRegression(class_weight='balanced', max_iter=500,
                               random_state=random_seed, solver='lbfgs')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n  Evaluation on held-out 15%:")
    print(classification_report(y_test, y_pred, target_names=['Low relevance', 'High relevance']))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model,   'models/hint_scorer.pkl')
    joblib.dump(ohe_vec, 'models/hint_ohe.pkl')
    print("\n  Saved: models/hint_scorer.pkl")
    print("  Saved: models/hint_ohe.pkl")
    print("=" * 55)
    print("  Done!")

if __name__ == "__main__":
    train_hint_scorer()
