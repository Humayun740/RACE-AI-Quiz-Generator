import os
import numpy as np
import joblib
from scipy.sparse import hstack

def _build_features(article: str, question: str, options: list, vectorizer):

    n = len(options)
    passages = [article] * n
    questions = [question] * n

    p_vecs = vectorizer.transform(passages)
    o_vecs = vectorizer.transform(options)
    qo_vecs = vectorizer.transform([q + " " + o for q, o in zip(questions, options)])

    sim_p_o = np.asarray(p_vecs.multiply(o_vecs).sum(axis=1)).squeeze()
    sim_p_qo = np.asarray(p_vecs.multiply(qo_vecs).sum(axis=1)).squeeze()

    overlap_ratios, opt_lengths = [], []
    for p, o in zip(passages, options):
        p_words = set(p.lower().split())
        o_words = set(o.lower().split())
        overlap_ratios.append(len(p_words & o_words) / max(len(o_words), 1))
        opt_lengths.append(len(o_words))

    dense = np.column_stack([sim_p_o, sim_p_qo, overlap_ratios, opt_lengths])
    return hstack([dense, o_vecs])

def predict_correct_option(article: str, question: str, options: list) -> dict | None:

    vec_path = 'models/tfidf_vectorizer_checkpoint.pkl'
    lr_path  = 'models/model_a_lr_checkpoint.pkl'
    svm_path = 'models/model_a_svm_checkpoint.pkl'

    if not all(os.path.exists(p) for p in [vec_path, lr_path, svm_path]):
        return None

    try:
        import sys
        import __main__
        sys.path.insert(0, 'src')
        import model_a_train_improved  
        if not hasattr(__main__, 'lemmatized_tokenizer'):
            __main__.lemmatized_tokenizer = model_a_train_improved.lemmatized_tokenizer
    except Exception:
        pass

    try:
        vectorizer = joblib.load(vec_path)
        model_lr   = joblib.load(lr_path)
        model_svm  = joblib.load(svm_path)

        X = _build_features(article, question, options, vectorizer)

        lr_probs   = model_lr.predict_proba(X)[:, 1]
        svm_scores = model_svm.decision_function(X)

        best_lr  = int(np.argmax(lr_probs))
        best_svm = int(np.argmax(svm_scores))

        svm_min, svm_max = svm_scores.min(), svm_scores.max()
        svm_norm = ((svm_scores - svm_min) / max(svm_max - svm_min, 1e-9)).tolist()

        return {
            'lr_predicted':  options[best_lr],
            'svm_predicted': options[best_svm],
            'lr_scores':  {opt: round(float(s), 4) for opt, s in zip(options, lr_probs)},
            'svm_scores': {opt: round(float(s), 4) for opt, s in zip(options, svm_norm)},
        }
    except Exception as e:
        print(f"[Model A inference error]: {e}")
        return None
