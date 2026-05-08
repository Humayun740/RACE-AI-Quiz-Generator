import joblib
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from tokenizer_utils import lemmatized_tokenizer

def generate_distractors(article, correct_answer, num_distractors=3):
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    sentences = nltk.sent_tokenize(article)
    
    all_vectors = vectorizer.transform(sentences + [correct_answer])
    sentence_vectors = all_vectors[:-1]
    answer_vector = all_vectors[-1]
    
    scores = cosine_similarity(answer_vector, sentence_vectors).flatten()
    
    ranked_indices = scores.argsort()[::-1]
    
    distractors = []
    for idx in ranked_indices:
        candidate = sentences[idx].strip()
        if candidate.lower() not in correct_answer.lower() and len(candidate.split()) > 3:
            if candidate not in distractors:
                distractors.append(candidate)
        
        if len(distractors) >= num_distractors:
            break
            
    while len(distractors) < num_distractors:
        distractors.append("None of the above.")
        
    return distractors