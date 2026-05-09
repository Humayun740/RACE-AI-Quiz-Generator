import pandas as pd
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import os

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
    for _, row in df.iterrows():
        ans_col = row['answer']
        article = str(row['article'])
        question = str(row['question'])
        for opt in ['A', 'B', 'C', 'D']:
            label = 1 if opt == ans_col else 0
            text_content = f"passage: {article} [SEP] question: {question} [SEP] option: {row[opt]}"
            expanded_data.append({'text': text_content, 'label': label})
    
    edf = pd.DataFrame(expanded_data)
    print(f"Expanded Data: {len(edf)} rows.")

    train_df, temp_df = train_test_split(edf, test_size=0.2, random_state=42, stratify=edf['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    os.makedirs('data/processed', exist_ok=True)
    test_df.to_csv('data/processed/sample_test.csv', index=False)

    print("Vectorizing full dataset (30,000 Features)...")
    vectorizer = TfidfVectorizer(
        tokenizer=lemmatized_tokenizer,
        stop_words=lemmatized_stop_words,
        ngram_range=(1, 2),
        max_features=30000,
        min_df=5,
        strip_accents='unicode'
    )
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])

    print("Training Logistic Regression (C=0.5, Balanced)...")
    model_lr = LogisticRegression(
        C=0.5, 
        max_iter=3000, 
        class_weight='balanced', 
        solver='saga', 
        n_jobs=-1
    ).fit(X_train, train_df['label'])
    joblib.dump(model_lr, 'models/model_a_lr_checkpoint.pkl')

    print("Training Linear SVM (C=0.1, Balanced)...")
    model_svm = LinearSVC(
        C=0.1,
        class_weight='balanced', 
        max_iter=3000,
        dual=False
    ).fit(X_train, train_df['label'])
    joblib.dump(model_svm, 'models/model_a_svm_checkpoint.pkl')
    
    joblib.dump(vectorizer, 'models/tfidf_vectorizer_checkpoint.pkl')

    print("\n" + "="*40)
    print("FINAL EVALUATION: LOGISTIC REGRESSION")
    print("="*40)
    y_pred_lr = model_lr.predict(X_val)
    print(classification_report(val_df['label'], y_pred_lr))
    print("Confusion Matrix:")
    print(confusion_matrix(val_df['label'], y_pred_lr))

    print("\n" + "="*40)
    print("FINAL EVALUATION: LINEAR SVM")
    print("="*40)
    y_pred_svm = model_svm.predict(X_val)
    print(classification_report(val_df['label'], y_pred_svm))
    print("Confusion Matrix:")
    print(confusion_matrix(val_df['label'], y_pred_svm))

    print("\nDone! Checkpoints saved in models/ folder.")

if __name__ == "__main__":
    train_pipeline()