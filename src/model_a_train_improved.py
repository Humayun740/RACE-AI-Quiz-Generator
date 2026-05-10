import pandas as pd
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import os
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction import text
base_stop_words = text.ENGLISH_STOP_WORDS
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
    sim_p_o = np.asarray(p_vecs.multiply(o_vecs).sum(axis=1)).squeeze()
    sim_p_qo = np.asarray(p_vecs.multiply(qo_vecs).sum(axis=1)).squeeze()
    
    print("  -> Calculating word overlaps...")
    overlap_ratios = []
    opt_lengths = []
    for p, o in zip(passages, options):
        p_words = set(p.lower().split())
        o_words = set(o.lower().split())
        
        if not o_words:
            overlap_ratios.append(0.0)
        else:
            overlap_ratios.append(len(p_words.intersection(o_words)) / len(o_words))
        opt_lengths.append(len(o_words))
        
    X = np.column_stack([
        sim_p_o, 
        sim_p_qo, 
        overlap_ratios, 
        opt_lengths
    ])
    
    from scipy.sparse import hstack
    X_combined = hstack([X, o_vecs])
    
    return X_combined


def train_pipeline():
    print("Loading full RACE dataset...")
    input_path = 'data/raw/train.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please place train.csv in data/raw/")
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
    val_q, test_q = train_test_split(temp_q, test_size=0.5, random_state=42)
    
    train_df = edf[edf['q_id'].isin(train_q)]
    val_df = edf[edf['q_id'].isin(val_q)]
    test_df = edf[edf['q_id'].isin(test_q)]
    
    os.makedirs('data/processed', exist_ok=True)
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
    X_val = extract_features(val_df, vectorizer, is_train=False)

    print("Training Linear SVM...")
    model_svm = LinearSVC(
        C=0.1,
        class_weight='balanced', 
        max_iter=3000,
        dual=False
    ).fit(X_train, train_df['label'])
    joblib.dump(model_svm, 'models/model_a_svm_checkpoint.pkl')

    print("Training Logistic Regression (For Comparison)...")
    model_lr = LogisticRegression(
        C=0.5, 
        max_iter=3000, 
        class_weight='balanced', 
        solver='saga', 
        n_jobs=-1
    ).fit(X_train, train_df['label'])
    joblib.dump(model_lr, 'models/model_a_lr_checkpoint.pkl')
    
    joblib.dump(vectorizer, 'models/tfidf_vectorizer_checkpoint.pkl')

    print("\n" + "="*40)
    print("FINAL EVALUATION: LINEAR SVM")
    print("="*40)
    y_pred_svm = model_svm.predict(X_val)
    print(classification_report(val_df['label'], y_pred_svm))
    print("Confusion Matrix:")
    print(confusion_matrix(val_df['label'], y_pred_svm))

    print("\n" + "="*40)
    print("FINAL EVALUATION: LOGISTIC REGRESSION")
    print("="*40)
    y_pred_lr = model_lr.predict(X_val)
    print(classification_report(val_df['label'], y_pred_lr))
    print("Confusion Matrix:")
    print(confusion_matrix(val_df['label'], y_pred_lr))

    print("\n" + "="*40)
    print("MULTIPLE CHOICE ACCURACY EVALUATION")
    print("="*40)
    
    val_df = val_df.copy()
    val_df['prob_lr'] = model_lr.predict_proba(X_val)[:, 1]
    val_df['score_svm'] = model_svm.decision_function(X_val)
    
    correct_lr = 0
    correct_svm = 0
    total = 0
    for q_id, group in val_df.groupby('q_id'):
        best_lr_idx = group['prob_lr'].idxmax()
        if group.loc[best_lr_idx, 'label'] == 1:
            correct_lr += 1
            
        best_svm_idx = group['score_svm'].idxmax()
        if group.loc[best_svm_idx, 'label'] == 1:
            correct_svm += 1
            
        total += 1
        
    print(f"Logistic Regression True MC Accuracy: {correct_lr / total:.2%}")
    print(f"Linear SVM True MC Accuracy: {correct_svm / total:.2%}")

    print("\nDone! Checkpoints saved in models/ folder.")

if __name__ == "__main__":
    train_pipeline()
