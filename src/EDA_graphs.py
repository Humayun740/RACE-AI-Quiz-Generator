import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')

def generate_eda():
    print("=" * 60)
    print("  RACE DATASET — COMPREHENSIVE EDA REPORT")
    print("=" * 60)

    df = pd.read_csv('data/raw/train.csv', nrows=10000)
    os.makedirs('report/images', exist_ok=True)

    print("\n[1] DATA OVERVIEW")
    print(f"  Shape     : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns   : {list(df.columns)}")
    print(f"\n  Data Types:\n{df.dtypes.to_string()}")
    print(f"\n  Head (3 rows):\n{df.head(3).to_string()}")

    print("\n[2] MISSING VALUE ANALYSIS")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Count': missing, 'Percent (%)': missing_pct})
    if missing.any():
        print(missing_df[missing_df['Count'] > 0].to_string())
    else:
        print("  No missing values found in any column.")

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Value Heatmap (10,000 samples)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report/images/missing_values.png', dpi=120, bbox_inches='tight')
    plt.close()

    df['article_len'] = df['article'].fillna("").apply(lambda x: len(x.split()))
    df['question_len'] = df['question'].fillna("").apply(lambda x: len(x.split()))
    df['A_len'] = df['A'].fillna("").apply(lambda x: len(x.split()))
    df['B_len'] = df['B'].fillna("").apply(lambda x: len(x.split()))
    df['C_len'] = df['C'].fillna("").apply(lambda x: len(x.split()))
    df['D_len'] = df['D'].fillna("").apply(lambda x: len(x.split()))

    print("\n[3] STATISTICAL ANALYSIS (Text Length Features)")
    stat_cols = ['article_len', 'question_len', 'A_len', 'B_len', 'C_len', 'D_len']
    stats_df = df[stat_cols].describe().round(2)
    print(stats_df.to_string())

    print("\n[4] OUTLIER DETECTION (Article Lengths)")
    Q1 = df['article_len'].quantile(0.25)
    Q3 = df['article_len'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_outliers = df[(df['article_len'] < lower) | (df['article_len'] > upper)]
    z_scores = np.abs(stats.zscore(df['article_len']))
    z_outliers = (z_scores > 3).sum()
    print(f"  Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}")
    print(f"  Lower fence: {lower:.0f}  |  Upper fence: {upper:.0f}")
    print(f"  IQR outliers : {len(iqr_outliers)} ({len(iqr_outliers)/len(df)*100:.2f}%)")
    print(f"  Z>3 outliers : {z_outliers} ({z_outliers/len(df)*100:.2f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order_counts = df['answer'].value_counts().reindex(['A', 'B', 'C', 'D'])
    bars = axes[0].bar(['A', 'B', 'C', 'D'], order_counts.values,
                       color=['#4F46E5', '#EC4899', '#10B981', '#F59E0B'], edgecolor='white')
    for bar, v in zip(bars, order_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 20, str(v),
                     ha='center', fontweight='bold', fontsize=10)
    axes[0].set_title('Distribution of Correct Answers (A–D)', fontweight='bold')
    axes[0].set_xlabel('Correct Option')
    axes[0].set_ylabel('Count')

    axes[1].hist(df['article_len'], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1].axvline(df['article_len'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f"Mean: {df['article_len'].mean():.0f}")
    axes[1].axvline(df['article_len'].median(), color='orange', linestyle='--',
                    linewidth=2, label=f"Median: {df['article_len'].median():.0f}")
    axes[1].axvline(upper, color='darkred', linestyle=':', linewidth=1.5,
                    label=f"IQR Upper: {upper:.0f}")
    axes[1].set_title('Article Length Distribution (Word Count)', fontweight='bold')
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('report/images/answer_dist_and_article_len.png', dpi=120, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(df['question_len'], bins=25, color='#EC4899', edgecolor='white', alpha=0.85)
    axes[0].set_title('Question Length Distribution', fontweight='bold')
    axes[0].set_xlabel('Number of Words')
    axes[0].set_ylabel('Frequency')

    opt_melt = pd.melt(df[['A_len', 'B_len', 'C_len', 'D_len']],
                       var_name='Option', value_name='Length')
    sns.boxplot(x='Option', y='Length', data=opt_melt,
                palette=['#4F46E5', '#EC4899', '#10B981', '#F59E0B'], ax=axes[1])
    axes[1].set_title('Answer Option Length Distribution (A–D)', fontweight='bold')
    axes[1].set_xlabel('Option')
    axes[1].set_ylabel('Word Count')
    plt.tight_layout()
    plt.savefig('report/images/question_option_lengths.png', dpi=120, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bp = axes[0].boxplot(df['article_len'], patch_artist=True,
                         boxprops=dict(facecolor='lightblue'),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(color='gray'),
                         flierprops=dict(marker='o', color='darkred', alpha=0.3, markersize=3))
    axes[0].set_title('Article Length — Box Plot (IQR Outlier Detection)', fontweight='bold')
    axes[0].set_ylabel('Word Count')
    axes[0].set_xticks([])

    axes[1].hist(z_scores, bins=50, color='mediumpurple', edgecolor='white', alpha=0.8)
    axes[1].axvline(3, color='red', linestyle='--', linewidth=2, label='|Z| = 3 threshold')
    axes[1].set_title('Z-Score Distribution of Article Lengths', fontweight='bold')
    axes[1].set_xlabel('|Z-Score|')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('report/images/outlier_detection.png', dpi=120, bbox_inches='tight')
    plt.close()

    print("\n[5] FEATURE CORRELATION ANALYSIS (sample of 1000 rows)")
    sample = df.sample(min(1000, len(df)), random_state=42).copy()

    rows = []
    for _, row in sample.iterrows():
        for opt in ['A', 'B', 'C', 'D']:
            label = 1 if row['answer'] == opt else 0
            art = str(row['article'])
            q = str(row['question'])
            o = str(row[opt])
            art_words = set(art.lower().split())
            opt_words = set(o.lower().split())
            q_words = set(q.lower().split())
            rows.append({
                'label': label,
                'art_opt_overlap': len(art_words & opt_words) / max(len(opt_words), 1),
                'q_opt_overlap': len(q_words & opt_words) / max(len(opt_words), 1),
                'opt_len': len(opt_words),
                'art_len': len(art_words),
                'q_len': len(q_words),
            })

    feat_df = pd.DataFrame(rows)
    corr = feat_df.corr()
    print("\n  Pearson Correlation with label (1=correct):")
    print(corr[['label']].round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title('Feature Correlation Heatmap', fontweight='bold')

    feat_melt = feat_df.melt(id_vars='label',
                             value_vars=['art_opt_overlap', 'q_opt_overlap'],
                             var_name='Feature', value_name='Value')
    feat_melt['label'] = feat_melt['label'].map({0: , 1: })
    sns.boxplot(x='Feature', y='Value', hue='label', data=feat_melt,
                palette={'Incorrect (0)': , 'Correct (1)': }, ax=axes[1])
    axes[1].set_title('Feature Values by Answer Correctness', fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Overlap Ratio')
    plt.tight_layout()
    plt.savefig('report/images/correlation_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()

    print("\n[6] EDA COMPLETE — Files saved to report/images/:")
    print("  - missing_values.png")
    print("  - answer_dist_and_article_len.png")
    print("  - question_option_lengths.png")
    print("  - outlier_detection.png")
    print("  - correlation_analysis.png")
    print("=" * 60)

if __name__ == "__main__":
    generate_eda()
