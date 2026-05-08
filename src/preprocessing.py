import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def process_set(df):
    """Expands articles into the (Article + Question + Option) format for Model A."""
    data = []
    for _, row in df.iterrows():
        for opt in ['A', 'B', 'C', 'D']:
            is_correct = 1 if row['answer'] == opt else 0
            text = f"{row['article']} {row['question']} {row[opt]}"
            data.append({'text': text, 'label': is_correct})
    return pd.DataFrame(data)

def run_preprocessing():
    input_path = 'data/raw/train.csv'
    if not os.path.exists(input_path):
        print("Error: train.csv not found in data/raw/")
        return

    print("Loading 10,000 rows and performing manual 80-10-10 split...")
    full_df = pd.read_csv(input_path, nrows=10000)

    train_raw, temp_raw = train_test_split(full_df, test_size=0.20, random_state=42)
    val_raw, test_raw = train_test_split(temp_raw, test_size=0.50, random_state=42)

    print("Expanding datasets for verification task...")
    train_expanded = process_set(train_raw)
    val_expanded = process_set(val_raw)
    test_expanded = process_set(test_raw)

    print("Vectorizing text (fitting on train set only)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, sublinear_tf=True)
    
    X_train = vectorizer.fit_transform(train_expanded['text'])
    X_val = vectorizer.transform(val_expanded['text'])
    X_test = vectorizer.transform(test_expanded['text'])

    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    joblib.dump((X_train, train_expanded['label']), 'data/processed/train_data.pkl')
    joblib.dump((X_val, val_expanded['label']), 'data/processed/val_data.pkl')
    joblib.dump((X_test, test_expanded['label']), 'data/processed/test_data.pkl')
    
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    test_raw.head(10).to_csv('data/processed/sample_test.csv', index=False)
    
    print(f"Done! Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

if __name__ == "__main__":
    run_preprocessing()