import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_distractors(article, correct_answer, n=3):
    
    sentences = [s.strip() for s in article.split('.') if len(s.split()) > 3]
    
    
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    
    ans_vec = vectorizer.transform([correct_answer])
    sent_vecs = vectorizer.transform(sentences)
    
    
    scores = cosine_similarity(ans_vec, sent_vecs).flatten()
    
    
    
    ranked_idx = scores.argsort()[::-1]
    
    distractors = []
    for idx in ranked_idx:
        candidate = sentences[idx]
        
        if scores[idx] < 0.9 and candidate.lower() != correct_answer.lower():
            distractors.append(candidate)
        if len(distractors) >= n:
            break
            
            
    while len(distractors) < n:
        distractors.append("Information not mentioned in text")
        
    return distractors

if __name__ == "__main__":
    
    art = "The solar system has eight planets. Mars is known as the Red Planet."
    ans = "Mars"
    print(f"Generated Distractors: {generate_distractors(art, ans)}")