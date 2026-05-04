#cc cc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_eda():
    print("Generating EDA visualizations for your report...")
    df = pd.read_csv('data/raw/train.csv', nrows=10000)
    
    # Create a folder for report images
    os.makedirs('report/images', exist_ok=True)

    # 1. Answer Distribution (Are the answers balanced?)
    plt.figure(figsize=(8, 5))
    sns.countplot(x='answer', data=df, palette='viridis')
    plt.title('Distribution of Correct Answers (A, B, C, D)')
    plt.savefig('report/images/answer_dist.png')
    plt.close()

    # 2. Article Length Distribution
    df['article_len'] = df['article'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['article_len'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Article Lengths (Word Count)')
    plt.xlabel('Number of Words')
    plt.savefig('report/images/article_length.png')
    plt.close()

    print("Done! Charts saved in report/images/")
    print("You can use these in your final report to show you analyzed the data.")

if __name__ == "__main__":
    generate_eda()