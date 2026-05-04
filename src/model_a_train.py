import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score

def train_model_a():
    # 1. Load Data
    if not os.path.exists('data/processed/X_train.pkl'):
        print("Error: Vectorized data not found. Run preprocessing.py first!")
        return

    print("Loading processed data...")
    X = joblib.load('data/processed/X_train.pkl')
    y = joblib.load('data/processed/y_train.pkl')

    # 2. Train Logistic Regression (Supervised Model 1)
    print("Training Logistic Regression (Model 1)...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    joblib.dump(lr, 'models/model_a_lr.pkl')

    # 3. Train Linear SVM (Supervised Model 2 - for extra marks)
    print("Training Linear SVM (Model 2)...")
    svm = LinearSVC(max_iter=1000)
    svm.fit(X, y)
    joblib.dump(svm, 'models/model_a_svm.pkl')

    # 4. K-Means (Unsupervised Requirement - 20 Marks)
    print("Running K-Means Clustering (Unsupervised Task)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Calculate Silhouette Score (Metric for unsupervised)
    # Using a sample of 1000 to keep it fast on your i7
    score = silhouette_score(X, clusters, sample_size=1000)

    print("\n" + "="*30)
    print("DAY 1 RESULTS")
    print("="*30)
    print(f"LR Training Accuracy: {lr.score(X, y):.4f}")
    print(f"SVM Training Accuracy: {svm.score(X, y):.4f}")
    print(f"K-Means Silhouette Score: {score:.4f}")
    print("="*30)
    print("All Model A scripts complete. Check the 'models' folder for your .pkl files.")

if __name__ == "__main__":
    train_model_a()