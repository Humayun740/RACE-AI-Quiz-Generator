import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_distractors(article, correct_answer, n=3):
    #developer dashboard with performance metrics and EDA images(ignore this comment)
    # 1. Split article into sentencs
    sentences = [s.strip() for s in article.split('.') if len(s.split()) > 3]
    
    # 2. Load the vectorizer you fit yesterday
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    # 3. Vectorize answer and sentences
    ans_vec = vectorizer.transform([correct_answer])
    sent_vecs = vectorizer.transform(sentences)
    
    # 4. Calculate similarity
    scores = cosine_similarity(ans_vec, sent_vecs).flatten()
    
    # 5. Pick "Plausible" sentences
    # We want sentences that are similar but not identical
    ranked_idx = scores.argsort()[::-1]
    
    distractors = []
    for idx in ranked_idx:
        candidate = sentences[idx]
        # Ignore if it's too similar to the actual answer
        if scores[idx] < 0.9 and candidate.lower() != correct_answer.lower():
            distractors.append(candidate)
        if len(distractors) >= n:
            break
            
            
    while len(distractors) < n:
        distractors.append("Information not mentioned in text")
        
    return distractors

if __name__ == "__main__":
    # Test sample
    art = "The solar system has eight planets. Mars is known as the Red Planet."
    ans = "Mars"
    print(f"Generated Distractors: {generate_distractors(art, ans)}")