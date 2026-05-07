import pandas as pd
from distraction_gen import generate_distractors
from hint_gen import get_graduated_hints

def run_test():
    # load sample test data   
    try:
        test_df = pd.read_csv('data/processed/sample_test.csv')
    except:
        test_df = pd.read_csv('data/raw/train.csv', nrows=5)

    print("--- MODEL B EVALUATION TEST ---")
    
    for i, row in test_df.iterrows():
        article = row['article']
        question = row['question']
        answer_text = row[row['answer']] 
        
        print(f"\nSample {i+1}:")
        print(f"Question: {question}")
        print(f"Correct Answer: {answer_text}")
        
        
        distractors = generate_distractors(article, answer_text)
        print(f"Generated Distractors: {distractors}")
        
        
        hints = get_graduated_hints(article, question, answer_text)
        print(f"Graduated Hints:")
        for level, text in hints.items():
            print(f"  - {level}: {text[:100]}...") 

if __name__ == "__main__":
    run_test()