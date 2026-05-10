import joblib
import nltk
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def _word_jaccard(a: str, b: str) -> float:

    sa, sb = set(a.lower().split()), set(b.lower().split())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0

def _char_jaccard(a: str, b: str) -> float:

    sa, sb = set(a.lower()), set(b.lower())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0

def _pos_pattern(text: str) -> str:

    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)

        def coarsen(tag):
            if tag.startswith('NN'): return 'N'
            if tag.startswith('VB'): return 'V'
            if tag.startswith('JJ'): return 'J'
            if tag == 'DT':          return 'D'
            if tag == 'CD':          return 'C'
            return 'X'
        return ''.join(coarsen(t) for _, t in tagged)
    except Exception:
        return ''

def _pos_compatible(candidate: str, correct_answer: str) -> bool:

    ans_words = len(correct_answer.split())
    cand_words = len(candidate.split())
    if ans_words <= 2 and cand_words <= 3:
        return True

    ans_pat  = _pos_pattern(correct_answer)
    cand_pat = _pos_pattern(candidate)
    if not ans_pat or not cand_pat:
        return True  

    return ans_pat[0] == cand_pat[0] if (ans_pat and cand_pat) else True

def _extract_noun_phrases(article: str) -> list:

    sentences = nltk.sent_tokenize(article)
    candidates = set()
    grammar = r"NP: {<DT>?<JJ.*>*<NN.*>+|<NNP>+|<CD>+}"
    chunker = nltk.RegexpParser(grammar)
    for sent in sentences:
        try:
            tokens = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(tokens)
            tree = chunker.parse(tagged)
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                phrase = " ".join(word for word, pos in subtree.leaves())
                if 1 <= len(phrase.split()) <= 6 and len(phrase) > 2:
                    candidates.add(phrase.strip())
        except Exception:
            words = sent.split()
            for w in words:
                if len(w) > 4 and w[0].isupper():
                    candidates.add(w)
    return list(candidates)

def _frequency_candidates(article: str, top_n: int = 20) -> list:

    stop = {
        'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'has', 'have',
        'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may',
        'might', 'can', 'could', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
        'by', 'from', 'that', 'this', 'it', 'he', 'she', 'they', 'we', 'i',
        'his', 'her', 'their', 'its', 'and', 'or', 'but', 'so', 'not', 'no',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', article.lower())
    freq = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]

def is_valid_distractor(candidate: str, correct_answer: str, seen: list) -> bool:

    cand = candidate.strip()
    if len(cand) < 2:
        return False
    ans_lower = correct_answer.lower().strip()
    cand_lower = cand.lower()
    if cand_lower == ans_lower:
        return False
    if _word_jaccard(cand, correct_answer) > 0.8:
        return False

    for chosen in seen:
        if _word_jaccard(cand, chosen) > 0.6:
            return False
    return True

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

def _rank_with_tfidf(candidates: list, correct_answer: str, article: str, num: int) -> list:

    try:
        try:
            vectorizer = joblib.load('models/distractor_tfidf.pkl')
        except Exception:
            vectorizer = joblib.load('models/tfidf_vectorizer_checkpoint.pkl')

        all_vecs  = vectorizer.transform(candidates + [correct_answer])
        cand_vecs = all_vecs[:-1]
        ans_vec   = all_vecs[-1]
        scores    = cosine_similarity(ans_vec, cand_vecs).flatten()
        ranked    = scores.argsort()[::-1]

        distractors = []
        for idx in ranked:
            c = candidates[idx]
            if is_valid_distractor(c, correct_answer, distractors):
                distractors.append(c)
            if len(distractors) >= num:
                break
        return distractors
    except Exception as e:
        print(f"[distractor] TF-IDF ranking failed: {e}")
        return []

def generate_distractors(article: str, correct_answer: str,
                         num_distractors: int = 3) -> list:

    np_candidates   = _extract_noun_phrases(article)
    freq_candidates = _frequency_candidates(article, top_n=20)

    all_candidates = list(dict.fromkeys(np_candidates + freq_candidates))

    phrase_candidates = [
        c for c in all_candidates
        if is_valid_distractor(c, correct_answer, [])
    ]

    syntactic_matches = [c for c in phrase_candidates if _pos_compatible(c, correct_answer)]
    if len(syntactic_matches) >= num_distractors:
        phrase_candidates = syntactic_matches

    ranker_path = 'models/distractor_ranker.pkl'
    ohe_path    = 'models/distractor_ohe.pkl'
    tfidf_path  = 'models/distractor_tfidf.pkl'

    if all(os.path.exists(p) for p in [ranker_path, ohe_path, tfidf_path]) and phrase_candidates:
        try:
            model    = joblib.load(ranker_path)
            ohe_vec  = joblib.load(ohe_path)
            tfidf_vec = joblib.load(tfidf_path)

            X = np.array([
                compute_features(c, correct_answer, article, ohe_vec, tfidf_vec)
                for c in phrase_candidates
            ])

            expected_features = model.n_features_in_
            if X.shape[1] != expected_features:
                raise ValueError(f"Feature mismatch: got {X.shape[1]}, expected {expected_features}")

            scores = model.predict_proba(X)[:, 1]
            ranked = scores.argsort()[::-1]

            distractors = []
            for idx in ranked:
                c = phrase_candidates[idx]
                if is_valid_distractor(c, correct_answer, distractors):
                    distractors.append(c)
                if len(distractors) >= num_distractors:
                    break

            if len(distractors) >= num_distractors:
                return distractors

            extras = _rank_with_tfidf(phrase_candidates, correct_answer, article, num_distractors + 3)
            for e in extras:
                if is_valid_distractor(e, correct_answer, distractors):
                    distractors.append(e)
                if len(distractors) >= num_distractors:
                    return distractors

            if len(distractors) >= num_distractors:
                return distractors

        except Exception as e:
            print(f"[distractor] Ranker failed: {e}, using TF-IDF")

    distractors = _rank_with_tfidf(phrase_candidates, correct_answer, article, num_distractors)

    _pad(distractors, num_distractors)
    return distractors

def _pad(distractors: list, target: int) -> None:
    generics = ["None of the above", "All of the above", "Cannot be determined"]
    for g in generics:
        if len(distractors) >= target:
            break
        if g not in distractors:
            distractors.append(g)
    while len(distractors) < target:
        distractors.append("None of the above")
