import joblib
from sklearn.metrics.pairwise import cosine_similarity

def get_graduated_hints(article, question, correct_answer):
    sentences = [s.strip() for s in article.split('.') if len(s.split()) > 4]
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    target_text = f"{question} {correct_answer}"
    target_vec = vectorizer.transform([target_text])
    sent_vecs = vectorizer.transform(sentences)
    
    scores = cosine_similarity(target_vec, sent_vecs).flatten()
    ranked_idx = scores.argsort()
    
    hint_1 = sentences[ranked_idx[len(ranked_idx)//4]]
    hint_2 = sentences[ranked_idx[len(ranked_idx)//2]]
    hint_3 = sentences[ranked_idx[-1]]
    
    return {"Hint 1": hint_1, "Hint 2": hint_2, "Hint 3": hint_3}

if __name__ == "__main__":
    art = "Photosynthesis is how plants make food. They use sunlight, water, and carbon dioxide."
    que = "How do plants make food?"
    ans = "Photosynthesis"
    print(get_graduated_hints(art, que, ans))