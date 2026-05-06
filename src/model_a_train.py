import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score

def train_model_a():
    print("Loading the 80% manual training split...")
    X_train, y_train = joblib.load('data/processed/train_data.pkl')
    X_val, y_val = joblib.load('data/processed/val_data.pkl')

    # 1. Supervised: Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    joblib.dump(lr, 'models/model_a_lr.pkl')

    # 2. Supervised: SVM, this is a X value for LinearSVC, you can experiment with other kernels if you like
    print("Training Linear SVM...")
    svm = LinearSVC(max_iter=1000)
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'models/model_a_svm.pkl')

    # 3. Unsupervised: K-Means (20 Marks)
    print("Running K-Means on training data...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_train)
    score = silhouette_score(X_train, clusters, sample_size=500)

    print("\n" + "="*30)
    print("RESULTS (80-10-10 Split)")
    print("="*30)
    print(f"Validation Accuracy (LR):  {lr.score(X_val, y_val):.4f}")
    print(f"Validation Accuracy (SVM): {svm.score(X_val, y_val):.4f}")
    print(f"Silhouette Score:          {score:.4f}")
    print("="*30)

if __name__ == "__main__":
    train_model_a()