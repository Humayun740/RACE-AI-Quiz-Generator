import pandas as pd
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix,
                             silhouette_score, accuracy_score, f1_score)
import os
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction import text as sklearn_text
base_stop_words = sklearn_text.ENGLISH_STOP_WORDS
lemmatized_stop_words = [lemmatizer.lemmatize(w) for w in base_stop_words]

def lemmatized_tokenizer(text):
    tokens = nltk.word_tokenize(text.lower())
    return [lemmatizer.lemmatize(t) for t in tokens if t.isalnum()]

def extract_features(df, vectorizer, is_train=False):
    passages = df['article'].fillna("").astype(str).tolist()
    questions = df['question'].fillna("").astype(str).tolist()
    options = df['option'].fillna("").astype(str).tolist()

    if is_train:
        vectorizer.fit(passages + questions + options)

    print("  -> Vectorizing texts...")
    p_vecs = vectorizer.transform(passages)
    o_vecs = vectorizer.transform(options)
    qo_vecs = vectorizer.transform([q + " " + o for q, o in zip(questions, options)])

    print("  -> Calculating cosine similarities...")
    sim_p_o   = np.asarray(p_vecs.multiply(o_vecs).sum(axis=1)).squeeze()
    sim_p_qo  = np.asarray(p_vecs.multiply(qo_vecs).sum(axis=1)).squeeze()

    print("  -> Calculating word overlaps...")
    overlap_ratios, opt_lengths = [], []
    for p, o in zip(passages, options):
        p_words = set(p.lower().split())
        o_words = set(o.lower().split())
        overlap_ratios.append(len(p_words & o_words) / max(len(o_words), 1))
        opt_lengths.append(len(o_words))

    X_dense = np.column_stack([sim_p_o, sim_p_qo, overlap_ratios, opt_lengths])

    from scipy.sparse import hstack
    return hstack([X_dense, o_vecs])

def clustering_purity(true_labels, cluster_labels):

    n = len(true_labels)
    clusters = np.unique(cluster_labels)
    total_correct = 0
    for c in clusters:
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        labels_in_cluster = np.array(true_labels)[mask]
        majority_count = np.bincount(labels_in_cluster).max()
        total_correct += majority_count
    return total_correct / n

def print_comparison_table(results: dict):

    print("\n" + "=" * 68)
    print("  MODEL COMPARISON TABLE (Validation Set)")
    print("=" * 68)
    header = f"{'Model':<30} {'Accuracy':>10} {'F1 (wtd)':>10} {'MC Acc':>10}"
    print(header)
    print("-" * 68)
    for model_name, metrics in results.items():
        acc  = f"{metrics.get('accuracy', 0):.4f}"
        f1   = f"{metrics.get('f1_weighted', 0):.4f}"
        mc   = f"{metrics['mc_accuracy']:.2%}" if metrics.get('mc_accuracy') is not None else "  —"
        print(f"{model_name:<30} {acc:>10} {f1:>10} {mc:>10}")
    print("=" * 68)

def train_pipeline():
    print("Loading full RACE dataset...")
    input_path = 'data/raw/train.csv'

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Dataset Loaded. Total questions: {len(df)}")

    expanded_data = []
    print("Expanding dataset into binary classification format...")
    for idx, row in df.iterrows():
        ans_col = row['answer']
        for opt in ['A', 'B', 'C', 'D']:
            label = 1 if opt == ans_col else 0
            expanded_data.append({
                'q_id': idx,
                'article': row['article'],
                'question': row['question'],
                'option': row[opt],
                'label': label
            })

    edf = pd.DataFrame(expanded_data)
    print(f"Expanded Data: {len(edf)} rows.")

    q_ids = edf['q_id'].unique()
    train_q, temp_q = train_test_split(q_ids, test_size=0.2, random_state=42)
    val_q, test_q   = train_test_split(temp_q, test_size=0.5, random_state=42)

    train_df = edf[edf['q_id'].isin(train_q)]
    val_df   = edf[edf['q_id'].isin(val_q)]
    test_df  = edf[edf['q_id'].isin(test_q)]

    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    test_df.to_csv('data/processed/sample_test.csv', index=False)

    print("Vectorizing dataset and extracting engineered features...")
    vectorizer = TfidfVectorizer(
        tokenizer=lemmatized_tokenizer,
        stop_words=lemmatized_stop_words,
        ngram_range=(1, 2),
        max_features=5000,
        min_df=5,
        strip_accents='unicode'
    )

    print("Processing Training Data:")
    X_train = extract_features(train_df, vectorizer, is_train=True)
    print("Processing Validation Data:")
    X_val   = extract_features(val_df,   vectorizer, is_train=False)

    y_train = train_df['label'].values
    y_val   = val_df['label'].values

    results = {}

    print("\n" + "=" * 40)
    print("TRAINING: LINEAR SVM")
    print("=" * 40)
    model_svm = LinearSVC(C=0.1, class_weight='balanced', max_iter=3000, dual=False)
    model_svm.fit(X_train, y_train)
    joblib.dump(model_svm, 'models/model_a_svm_checkpoint.pkl')

    y_pred_svm = model_svm.predict(X_val)
    print(classification_report(y_val, y_pred_svm))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_svm))
    results['Linear SVM'] = {
        'accuracy':    accuracy_score(y_val, y_pred_svm),
        'f1_weighted': f1_score(y_val, y_pred_svm, average='weighted'),
        'mc_accuracy': None,
    }

    print("\n" + "=" * 40)
    print("TRAINING: LOGISTIC REGRESSION")
    print("=" * 40)
    model_lr = LogisticRegression(
        C=0.5, max_iter=3000, class_weight='balanced', solver='saga'
    )
    model_lr.fit(X_train, y_train)
    joblib.dump(model_lr, 'models/model_a_lr_checkpoint.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer_checkpoint.pkl')

    y_pred_lr = model_lr.predict(X_val)
    print(classification_report(y_val, y_pred_lr))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_lr))
    results['Logistic Regression'] = {
        'accuracy':    accuracy_score(y_val, y_pred_lr),
        'f1_weighted': f1_score(y_val, y_pred_lr, average='weighted'),
        'mc_accuracy': None,
    }

    print("\n" + "=" * 40)
    print("TRAINING: COMPLEMENT NAIVE BAYES")
    print("=" * 40)

    model_nb = ComplementNB(alpha=0.5)
    model_nb.fit(X_train, y_train)
    joblib.dump(model_nb, 'models/model_a_nb_checkpoint.pkl')

    y_pred_nb = model_nb.predict(X_val)
    print(classification_report(y_val, y_pred_nb))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_nb))
    results['Complement Naive Bayes'] = {
        'accuracy':    accuracy_score(y_val, y_pred_nb),
        'f1_weighted': f1_score(y_val, y_pred_nb, average='weighted'),
        'mc_accuracy': None,
    }

    print("\n" + "=" * 40)
    print("UNSUPERVISED MODEL: K-MEANS CLUSTERING")
    print("=" * 40)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train)
    cluster_labels_val = kmeans.predict(X_val)

    try:
        sil = silhouette_score(X_val, cluster_labels_val, sample_size=5000, random_state=42)
        print(f"Silhouette Score (sample=5000): {sil:.4f}")
    except Exception:
        sil = 0.0
        print("Silhouette score calculation failed.")

    purity = clustering_purity(y_val, cluster_labels_val)
    print(f"Clustering Purity            : {purity:.4f}")
    print(f"(Comparison) Supervised SVM Accuracy: {results['Linear SVM']['accuracy']:.4f}")
    print(f"(Comparison) Supervised LR  Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
    print("Note: K-Means works without labels; purity shows cluster label alignment.")

    joblib.dump(kmeans, 'models/model_a_kmeans_checkpoint.pkl')
    results['K-Means (Unsupervised)'] = {
        'accuracy':    purity,
        'f1_weighted': sil,
        'mc_accuracy': None,
        'note': f'Silhouette={sil:.4f}, Purity={purity:.4f}',
    }

    print("\n" + "=" * 40)
    print("ENSEMBLE: HARD VOTING (SVM + LR + NB)")
    print("Strategy: Majority vote across 3 classifiers")
    print("=" * 40)
    hard_ensemble = VotingClassifier(
        estimators=[
            ('svm', model_svm),
            ('lr',  model_lr),
            ('nb',  model_nb),
        ],
        voting='hard'
    )
    hard_ensemble.fit(X_train, y_train)
    y_pred_hard = hard_ensemble.predict(X_val)
    print(classification_report(y_val, y_pred_hard))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_hard))
    joblib.dump(hard_ensemble, 'models/model_a_ensemble_checkpoint.pkl')
    results['Hard Voting Ensemble'] = {
        'accuracy':    accuracy_score(y_val, y_pred_hard),
        'f1_weighted': f1_score(y_val, y_pred_hard, average='weighted'),
        'mc_accuracy': None,
    }

    print("\n" + "=" * 40)
    print("ENSEMBLE: SOFT VOTING (LR + NB)")
    print("Strategy: Average predicted probabilities (LR + NB)")
    print("=" * 40)
    soft_ensemble = VotingClassifier(
        estimators=[
            ('lr', model_lr),
            ('nb', model_nb),
        ],
        voting='soft'
    )
    soft_ensemble.fit(X_train, y_train)
    y_pred_soft = soft_ensemble.predict(X_val)
    print(classification_report(y_val, y_pred_soft))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_soft))
    joblib.dump(soft_ensemble, 'models/model_a_soft_ensemble_checkpoint.pkl')
    results['Soft Voting Ensemble'] = {
        'accuracy':    accuracy_score(y_val, y_pred_soft),
        'f1_weighted': f1_score(y_val, y_pred_soft, average='weighted'),
        'mc_accuracy': None,
    }

    print("\n" + "=" * 40)
    print("MULTIPLE CHOICE ACCURACY EVALUATION")
    print("=" * 40)

    val_df = val_df.copy()
    val_df['prob_lr']   = model_lr.predict_proba(X_val)[:, 1]
    val_df['prob_nb']   = model_nb.predict_proba(X_val)[:, 1]
    val_df['prob_soft'] = soft_ensemble.predict_proba(X_val)[:, 1]
    val_df['score_svm'] = model_svm.decision_function(X_val)

    mc_counts = {k: 0 for k in ['lr', 'nb', 'svm', 'soft']}
    total = 0
    for _, group in val_df.groupby('q_id'):
        for key, col in [('lr', 'prob_lr'), ('nb', 'prob_nb'),
                         ('svm', 'score_svm'), ('soft', 'prob_soft')]:
            if group.loc[group[col].idxmax(), 'label'] == 1:
                mc_counts[key] += 1
        total += 1

    mc_results = {k: mc_counts[k] / total for k in mc_counts}
    print(f"Logistic Regression True MC Accuracy : {mc_results['lr']:.2%}")
    print(f"Naive Bayes True MC Accuracy         : {mc_results['nb']:.2%}")
    print(f"Linear SVM True MC Accuracy          : {mc_results['svm']:.2%}")
    print(f"Soft Voting Ensemble True MC Accuracy: {mc_results['soft']:.2%}")

    results['Logistic Regression']['mc_accuracy'] = mc_results['lr']
    results['Complement Naive Bayes']['mc_accuracy'] = mc_results['nb']
    results['Linear SVM']['mc_accuracy'] = mc_results['svm']
    results['Soft Voting Ensemble']['mc_accuracy'] = mc_results['soft']

    print_comparison_table(results)

    best_mc = max(
        [(k, v['mc_accuracy']) for k, v in results.items() if v.get('mc_accuracy')],
        key=lambda x: x[1]
    )
    best_f1 = max(
        [(k, v['f1_weighted']) for k, v in results.items()
         if k not in ['K-Means (Unsupervised)']],
        key=lambda x: x[1]
    )
    print(f"\n  Best MC Accuracy : {best_mc[0]} ({best_mc[1]:.2%})")
    print(f"  Best F1 (wtd)    : {best_f1[0]} ({best_f1[1]:.4f})")
    print("\nDone! Checkpoints saved in models/ folder.")

if __name__ == "__main__":
    train_pipeline()
