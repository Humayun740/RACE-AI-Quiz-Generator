import pandas as pd
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tokenizer_utils import lemmatized_tokenizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

def train_pipeline():
    print("Loading data and preparing 80-10-10 split...")
    df = pd.read_csv('data/raw/train.csv', nrows=5000)
    
    expanded_data = []
    for _, row in df.iterrows():
        ans_col = row['answer']
        for opt in ['A', 'B', 'C', 'D']:
            label = 1 if opt == ans_col else 0
            expanded_data.append({
                'text': f"{row['article']} [SEP] {row['question']} [SEP] {row[opt]}",
                'label': label
            })
    
    edf = pd.DataFrame(expanded_data)
    
    _, raw_test = train_test_split(df, test_size=0.1, random_state=42)
    raw_test.to_csv('data/processed/sample_test.csv', index=False)

    train_df, temp_df = train_test_split(edf, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("Vectorizing text with Lemmatization and Bi-grams...")
    vectorizer = TfidfVectorizer(
        tokenizer=lemmatized_tokenizer,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    
    print("Training Supervised Models...")
    model_lr = LogisticRegression(max_iter=1000).fit(X_train, train_df['label'])
    model_svm = LinearSVC().fit(X_train, train_df['label'])
    
    print("Running K-Means Clustering...")
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X_train)
    sil_score = silhouette_score(X_train, kmeans.labels_, sample_size=1000)
    
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(model_lr, 'models/model_a_lr.pkl')
    joblib.dump(model_svm, 'models/model_a_svm.pkl')
    print(f"Checkpoints saved! K-Means Silhouette: {sil_score:.4f}")

if __name__ == "__main__":
    train_pipeline()