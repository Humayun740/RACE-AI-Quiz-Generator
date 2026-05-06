import streamlit as st
import pandas as pd
import joblib
import random
import os

# Import your Model B logic from Day 2
from distraction_gen import generate_distractors
from hint_gen import get_graduated_hints

# --- PAGE CONFIG ---
#collapsable article reference
st.set_page_config(
    page_title="RACE AI Quiz Generator",
    page_icon="🤖",
    layout="wide"
)

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_resources():
    # Loading the vectorizer and models trained on Day 1
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model_lr = joblib.load('models/model_a_lr.pkl')
    model_svm = joblib.load('models/model_a_svm.pkl')
    
    # Load sample data for selection
    df = pd.read_csv('data/raw/train.csv', nrows=100)
    return vectorizer, model_lr, model_svm, df

vectorizer, model_lr, model_svm, df = load_resources()

# --- INITIALIZE SESSION STATE ---
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0
if 'hint_level' not in st.session_state:
    st.session_state['hint_level'] = 1
if 'options' not in st.session_state:
    st.session_state['options'] = []

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Project Modules")
page = st.sidebar.radio("Navigate to:", ["Article Selection", "Interactive Quiz", "Developer Dashboard"])

# --- PAGE 1: ARTICLE SELECTION ---
if page == "Article Selection":
    st.title("Step 1: Select or Input Article")
    st.markdown("Choose a passage from the RACE dataset to generate a smart quiz.")

    # Selectbox remembers its state
    sample_idx = st.selectbox(
        "Select Article ID:", 
        range(len(df)), 
        index=st.session_state['selected_index']
    )

    # Update state if selection changes
    if sample_idx != st.session_state['selected_index']:
        st.session_state['selected_index'] = sample_idx
        st.session_state['hint_level'] = 1 # Reset hints for new article
        st.session_state['options'] = [] # Reset options

    selected_row = df.iloc[sample_idx]
    
    # Store critical data in session state
    st.session_state['article'] = selected_row['article']
    st.session_state['question'] = selected_row['question']
    st.session_state['correct_ans'] = selected_row[selected_row['answer']]

    st.subheader("Article Content")
    st.text_area("Passage Text:", st.session_state['article'], height=350)
    
    st.success(f"Article #{sample_idx} is locked in. Head to the 'Interactive Quiz' page!")

# --- PAGE 2: INTERACTIVE QUIZ ---
elif page == "Interactive Quiz":
    st.title("Step 2: Test Your Comprehension")
    
    if 'article' not in st.session_state:
        st.warning("No article found. Please go to 'Article Selection' first.")
    else:
        # UX FIX: Collapsible Reference Article
        with st.expander("View Reference Article", expanded=False):
            st.write(st.session_state['article'])

        st.divider()
        
        # Display Question
        st.subheader("Question")
        st.info(st.session_state['question'])

        # Generate Options (only once per article selection)
        if not st.session_state['options']:
            distractors = generate_distractors(st.session_state['article'], st.session_state['correct_ans'])
            opts = distractors + [st.session_state['correct_ans']]
            random.shuffle(opts)
            st.session_state['options'] = opts

        # User Input
        user_choice = st.radio("Choose the most accurate answer:", st.session_state['options'])

        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Submit Answer", use_container_width=True):
                if user_choice == st.session_state['correct_ans']:
                    st.success("Correct!")
                else:
                    st.error("Incorrect. Try using a hint.")

        with col2:
            # Graduated Hintt System (Model B)
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

# --- PAGE 3: DEVELOPER DASHBOARD ---
elif page == "Developer Dashboard":
    st.title("Project Analytics & Metrics")
    st.markdown("Detailed performance report for the implemented Machine Learning models.")

    # Model A Performance Metrics
    st.subheader("Model A: Supervised & Unsupervised Learning")
    m1, m2, m3 = st.columns(3)
    m1.metric("Logistic Regression", "75.00%")
    m2.metric("Linear SVM", "75.00%")
    m3.metric("K-Means Silhouette", "0.0056")

    st.divider()

    # Data Architecture
    st.subheader("Data Management (80-10-10 Split)")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Data Distribution:**")
        st.write("- **Total Samples Processed:** 40,000 labels")
        st.write("- **Training Set:** 32,000")
        st.write("- **Validation Set:** 4,000")
        st.write("- **Test Set:** 4,000")
    
    with c2:
        # Check if EDA images exist before trying to display
        if os.path.exists("report/images/answer_dist.png"):
            st.image("report/images/answer_dist.png", caption="Article Class Distribution")
        else:
            st.info("Run EDA_graphs.py to generate visual metrics.")

    st.sidebar.markdown("---")
    st.sidebar.caption("System Status: Operational")