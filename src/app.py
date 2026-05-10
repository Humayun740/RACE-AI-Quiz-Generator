import streamlit as st
import pandas as pd
import joblib
import random
import os

from distraction_gen import generate_distractors
from hint_gen import get_graduated_hints

st.set_page_config(
    page_title="RACE AI Quiz Generator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4F46E5, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-top: 20px;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 20px;
        padding: 15px;
        background: #FFFFFF;
        border-radius: 15px;
        box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    div.stButton > button {
        border-radius: 25px;
        transition: all 0.3s ease-in-out;
        border: 1px solid #4F46E5;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    }
    
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
    }
    </style>
""", unsafe_allow_html=True)

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

st.markdown("<h1 class='main-title'>RACE AI Quiz Generator</h1>", unsafe_allow_html=True)

page = st.radio(
    "Navigation", 
    ["Article Selection", "Custom Input", "Interactive Quiz", "Developer Dashboard"],
    horizontal=True,
    label_visibility="collapsed"
)

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

elif page == "Custom Input":
    st.title("Custom Input: Test Your Own Data")
    st.markdown("Enter an article, a question, and the correct answer manually to test the model.")
    
    with st.form("custom_input_form"):
        custom_article = st.text_area("Article Text:", height=200, help="Paste your custom passage here.")
        custom_question = st.text_input("Question:", help="Enter the question related to the passage.")
        custom_answer = st.text_input("Correct Answer:", help="Enter the expected correct answer.")
        
        submitted = st.form_submit_button("Load Custom Data")
        if submitted:
            if custom_article.strip() and custom_question.strip() and custom_answer.strip():
                st.session_state['article'] = custom_article.strip()
                st.session_state['question'] = custom_question.strip()
                st.session_state['correct_ans'] = custom_answer.strip()
                st.session_state['hint_level'] = 1 
                st.session_state['options'] = [] 
                st.success("Custom data loaded successfully! Head over to the 'Interactive Quiz' page.")
            else:
                st.error("Please fill in all fields (Article, Question, and Correct Answer) before submitting.")

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

    st.markdown("---")
    st.caption("System Status: Operational")