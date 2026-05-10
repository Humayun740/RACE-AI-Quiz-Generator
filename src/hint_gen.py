import os
import nltk
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

nltk.download('punkt',    quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',  quiet=True)

lemmatizer = WordNetLemmatizer()

def lemmatized_tokenizer(text):
    tokens = nltk.word_tokenize(text.lower())
    return [lemmatizer.lemmatize(t) for t in tokens if t.isalnum()]

def _keyword_overlap(sentence: str, query: str) -> float:
    stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on',
            'at', 'to', 'for', 'of', 'and', 'or', 'but', 'it', 'that'}
    s_words = {w for w in sentence.lower().split() if w not in stop and len(w) > 2}
    q_words = {w for w in query.lower().split()    if w not in stop and len(w) > 2}
    if not s_words or not q_words:
        return 0.0
    return len(s_words & q_words) / len(s_words | q_words)

def _extract_features_lr(sentence: str, query: str, position: int,
                          total: int, answer: str, ohe_vec) -> list:
    kw_overlap    = _keyword_overlap(sentence, query)
    sent_position = position / max(total - 1, 1)
    sent_length   = min(len(sentence.split()), 50) / 50.0
    try:
        vecs    = ohe_vec.transform([sentence, query])
        ohe_sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])
    except Exception:
        ohe_sim = 0.0
    ans_words    = set(answer.lower().split())
    sent_words   = set(sentence.lower().split())
    ans_contains = float(bool(ans_words & sent_words))
    return [kw_overlap, sent_position, sent_length, ohe_sim, ans_contains]

def _score_with_lr(sentences: list, question: str, correct_answer: str) -> list:
    model_path = 'models/hint_scorer.pkl'
    ohe_path   = 'models/hint_ohe.pkl'
    if not (os.path.exists(model_path) and os.path.exists(ohe_path)):
        return []
    try:
        model   = joblib.load(model_path)
        ohe_vec = joblib.load(ohe_path)
        query   = f"{question} {correct_answer}"
        total   = len(sentences)
        X = np.array([
            _extract_features_lr(s, query, i, total, correct_answer, ohe_vec)
            for i, s in enumerate(sentences)
        ])
        probs  = model.predict_proba(X)[:, 1]
        scored = sorted(zip(probs.tolist(), sentences), key=lambda x: x[0], reverse=True)
        return scored
    except Exception as e:
        print(f"[hint_gen] LR scorer failed: {e}")
        return []

def _score_with_ohe(sentences: list, question: str, correct_answer: str) -> list:
    if not sentences:
        return []
    target   = f"{question} {correct_answer} {correct_answer}"
    all_text = sentences + [target]
    try:
        ohe     = CountVectorizer(binary=True, tokenizer=lemmatized_tokenizer,
                                  token_pattern=None)
        vecs    = ohe.fit_transform(all_text)
        s_vecs  = vecs[:-1]
        t_vec   = vecs[-1]
        scores  = cosine_similarity(t_vec, s_vecs).flatten()
        return sorted(zip(scores.tolist(), sentences), key=lambda x: x[0], reverse=True)
    except Exception:
        return list(zip([0.0] * len(sentences), sentences))

def get_graduated_hints(article: str, question: str, correct_answer: str) -> dict:
    sentences = nltk.sent_tokenize(article)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    if len(sentences) < 3:
        while len(sentences) < 3:
            sentences.append(sentences[-1] if sentences else correct_answer)
    ranked = _score_with_lr(sentences, question, correct_answer)
    if not ranked:
        ranked = _score_with_ohe(sentences, question, correct_answer)
    n = len(ranked)
    hint_3 = ranked[0][1]
    mid_idx = max(1, n // 3)
    hint_2  = ranked[mid_idx][1]
    low_idx = min(max(2, n // 2), n - 1)
    hint_1  = ranked[low_idx][1]
    used = {hint_3}
    if hint_2 in used:
        for _, s in ranked[1:]:
            if s not in used:
                hint_2 = s
                break
    used.add(hint_2)
    if hint_1 in used:
        for _, s in reversed(ranked):
            if s not in used:
                hint_1 = s
                break
    return {
        "Hint 1": f"General context: {hint_1}",
        "Hint 2": f"Look closer here: {hint_2}",
        "Hint 3": f"Key evidence: {hint_3}",
    }

if __name__ == "__main__":
    art = (
        "A bumpkin went to a big city for the first time. "
        "He saw many tall buildings and busy streets. "
        "When he entered a hotel he saw an elevator for the first time. "
        "An old woman walked into the elevator and a young woman walked out. "
        "He thought the elevator was a magic room that transformed people."
    )
    que = "What did the bumpkin misunderstand about the elevator?"
    ans = "elevator"
    result = get_graduated_hints(art, que, ans)
    for k, v in result.items():
        print(f"{k}: {v}")
