import streamlit as st
import pandas as pd
import joblib
import random
import os

from distraction_gen import generate_distractors
from hint_gen import get_graduated_hints

st.set_page_config(
    page_title="RACE AI Quiz Generator",
    layout="wide"
)

@st.cache_resource
def load_resources():
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model_lr = joblib.load('models/model_a_lr.pkl')
    model_svm = joblib.load('models/model_a_svm.pkl')
    df = pd.read_csv('data/raw/train.csv', nrows=100)
    return vectorizer, model_lr, model_svm, df

vectorizer, model_lr, model_svm, df = load_resources()

if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0
if 'hint_level' not in st.session_state:
    st.session_state['hint_level'] = 1
if 'options' not in st.session_state:
    st.session_state['options'] = []

st.sidebar.title("Project Modules")
page = st.sidebar.radio("Navigate to:", ["Article Selection", "Interactive Quiz", "Developer Dashboard"])

if page == "Article Selection":
    st.title("Step 1: Select or Input Article")
    st.markdown("Choose a passage from the RACE dataset to generate a smart quiz.")

    sample_idx = st.selectbox(
        "Select Article ID:", 
        range(len(df)), 
        index=st.session_state['selected_index']
    )

    if sample_idx != st.session_state['selected_index']:
        st.session_state['selected_index'] = sample_idx
        st.session_state['hint_level'] = 1 
        st.session_state['options'] = [] 

    selected_row = df.iloc[sample_idx]
    
    st.session_state['article'] = selected_row['article']
    st.session_state['question'] = selected_row['question']
    st.session_state['correct_ans'] = selected_row[selected_row['answer']]

    st.subheader("Article Content")
    st.text_area("Passage Text:", st.session_state['article'], height=350)
    
    st.success(f"Article #{sample_idx} is locked in. Head to the 'Interactive Quiz' page!")

elif page == "Interactive Quiz":
    st.title("Step 2: Test Your Comprehension")
    
    if 'article' not in st.session_state:
        st.warning("No article found. Please go to 'Article Selection' first.")
    else:
        with st.expander("View Reference Article", expanded=False):
            st.write(st.session_state['article'])

        st.divider()
        
        st.subheader("Question")
        st.info(st.session_state['question'])

        if not st.session_state['options']:
            distractors = generate_distractors(st.session_state['article'], st.session_state['correct_ans'])
            opts = distractors + [st.session_state['correct_ans']]
            random.shuffle(opts)
            st.session_state['options'] = opts

        user_choice = st.radio("Choose the most accurate answer:", st.session_state['options'])

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Submit Answer", use_container_width=True):
                if user_choice == st.session_state['correct_ans']:
                    st.success("Correct!")
                else:
                    st.error("Incorrect. Try using a hint.")

        with col2:
            if st.button("Get Graduated Hint", use_container_width=True):
                hints = get_graduated_hints(
                    st.session_state['article'], 
                    st.session_state['question'], 
                    st.session_state['correct_ans']
                )
                
                level = st.session_state['hint_level']
                st.write(f"**Hint {level}/3:** {hints[f'Hint {level}']}")
                
                if level < 3:
                    st.session_state['hint_level'] += 1
                else:
                    st.caption("No more hints available.")

elif page == "Developer Dashboard":
    st.title("Model Performance & NLP Metrics")
    st.markdown("""
    This dashboard displays the evaluation of our **Model B Generation Task** using standard NLP metrics.
    """)

    st.subheader("NLP Evaluation (Distractor Generation)")
    m1, m2, m3 = st.columns(3)
    m1.metric("BLEU Score", "0.1126", help="Measures word overlap with ground truth")
    m2.metric("ROUGE-L Score", "0.1040", help="Measures longest common subsequence")
    m3.metric("METEOR Score", "0.1776", help="Measures semantic similarity")

    st.divider()

    st.subheader("Model Checkpointing & Persistence")
    st.info("""
    Pre-trained weights and vectors are stored in the following checkpoint files:
    """)
    st.code("""
    - models/tfidf_vectorizer.pkl (Feature Extractor Checkpoint)
    - models/model_a_svm.pkl      (Supervised Model Checkpoint)
    - models/model_a_lr.pkl       (Supervised Model Checkpoint)
    """)

    st.divider()

    st.subheader("Data Management (80-10-10 Split)")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Data Distribution:**")
        st.write("- **Total Samples Processed:** 40,000 labels")
        st.write("- **Training Set:** 32,000 (80%)")
        st.write("- **Validation Set:** 4,000 (10%)")
        st.write("- **Test Set:** 4,000 (10%)")
        
        with st.expander("Show Supervised Verification Stats"):
            st.write("SVM Validation Accuracy: 75.00%")
            st.write("K-Means Silhouette Score: 0.0056")
    
    with c2:
        if os.path.exists("report/images/answer_dist.png"):
            st.image("report/images/answer_dist.png", caption="Article Class Distribution")
        else:
            st.info("Visuals can be found in the report/images folder.")

    st.sidebar.markdown("---")
    st.sidebar.caption("System Status: Operational")