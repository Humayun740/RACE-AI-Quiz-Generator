import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()

def lemmatized_tokenizer(text):
    tokens = nltk.word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens]

def get_graduated_hints(article, question, correct_answer):
    sentences = nltk.sent_tokenize(article)
    sentences = [s.strip() for s in sentences if len(s.split()) > 4]
    
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer_checkpoint.pkl')
    except:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    target_text = f"{question} {correct_answer} {correct_answer}"
    target_vec = vectorizer.transform([target_text])
    sent_vecs = vectorizer.transform(sentences)
    
    scores = cosine_similarity(target_vec, sent_vecs).flatten()
    
    ranked_idx = scores.argsort()
    
    h1_idx = ranked_idx[len(ranked_idx)//4]
    hint_1 = sentences[h1_idx]
    
    h2_idx = ranked_idx[len(ranked_idx)//2]
    hint_2 = sentences[h2_idx]
    
    best_idx = ranked_idx[-1]
    hint_3 = sentences[best_idx]
    
    if hint_3 == hint_2 and len(ranked_idx) > 3:
        hint_3 = sentences[ranked_idx[-2]]
        
    return {
        "Hint 1": f"Think about this context: {hint_1}",
        "Hint 2": f"Consider this detail: {hint_2}",
        "Hint 3": f"Key Evidence: {hint_3}"
    }

if __name__ == "__main__":
    art = "A bumpkin went to a big city. He saw an elevator but thought it was a magic room that turned old women into young girls."
    que = "What did the bumpkin misunderstand?"
    ans = "How an elevator works"
    print(get_graduated_hints(art, que, ans))