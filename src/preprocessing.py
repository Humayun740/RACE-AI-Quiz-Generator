import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def run_preprocessing():
    # 1. Load 10k Subset (CPU Friendly)
    # Ensure this path matches where your train.csv is
    input_path = 'data/raw/train.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please place train.csv in data/raw/")
        return

    print("Loading 10,000 rows from RACE dataset...")
    df = pd.read_csv(input_path, nrows=10000)

    # 2. Prepare Feature Text
    # Model A needs to see: Article + Question + Option
    # We expand the 10k questions into 40k rows (one for each option A, B, C, D)
    processed_data = []
    print("Formatting text for Model A...")
    for _, row in df.iterrows():
        for opt in ['A', 'B', 'C', 'D']:
            # label 1 if this option is the correct answer, else 0
            is_correct = 1 if row['answer'] == opt else 0
            text = f"{row['article']} {row['question']} {row[opt]}"
            processed_data.append({'text': text, 'label': is_correct})

    pdf = pd.DataFrame(processed_data)

    # 3. TF-IDF Vectorization
    # max_features=5000 keeps the memory usage low on your i7
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, sublinear_tf=True)
    X = vectorizer.fit_transform(pdf['text'])
    y = pdf['label']

    # 4. Save for the next step
    # We save these in data/processed and models/
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(X, 'data/processed/X_train.pkl')
    joblib.dump(y, 'data/processed/y_train.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    print(f"Done! Saved vectorized data ({X.shape}) and vectorizer.")

if __name__ == "__main__":
    run_preprocessing()