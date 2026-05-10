import streamlit as st
import pandas as pd
import joblib
import random
import os
import re
import time
import nltk

from distraction_gen import generate_distractors
from hint_gen import get_graduated_hints
from model_a_inference import predict_correct_option

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

st.set_page_config(
    page_title="RACE AI Quiz Generator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-top: 10px;
        margin-bottom: 4px;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 24px;
    }
    .option-box {
        background: #F8FAFC;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 10px;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .option-correct {
        background: #DCFCE7;
        border-color: #16A34A;
        color: #15803D;
        font-weight: 600;
    }
    .option-wrong {
        background: #FEE2E2;
        border-color: #DC2626;
        color: #B91C1C;
    }
    div.stButton > button {
        border-radius: 10px;
        transition: all 0.2s ease;
        font-weight: 600;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    }
    .hint-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #1E3A5F;
        font-size: 1rem;
        line-height: 1.6;
    }
    .hint-box-2 {
        background: #FEF9C3;
        border-left: 4px solid #B45309;
        color: #451A03;
    }
    .hint-box-3 {
        background: #FFF1F2;
        border-left: 4px solid #BE123C;
        color: #3B0764;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .error-box {
        background: #FEF2F2;
        border: 1.5px solid #DC2626;
        border-radius: 10px;
        padding: 14px 18px;
        color: #7F1D1D;
        font-size: 0.97rem;
        margin-top: 8px;
    }

    p, li, .stMarkdown, label { font-size: 1rem !important; min-font-size: 16px; }

    div.stButton > button[kind="primary"] {
        background: #4338CA;
        color: #FFFFFF;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model_lr_path = 'models/model_a_lr_checkpoint.pkl'
    model_svm_path = 'models/model_a_svm_checkpoint.pkl'
    vec_path = 'models/tfidf_vectorizer_checkpoint.pkl'

    model_lr = joblib.load(model_lr_path) if os.path.exists(model_lr_path) else None
    model_svm = joblib.load(model_svm_path) if os.path.exists(model_svm_path) else None

    df = pd.read_csv('data/raw/train.csv', nrows=200)
    return model_lr, model_svm, df

model_lr, model_svm, df = load_resources()

_missing_models = [
    p for p in [
        'models/model_a_lr_checkpoint.pkl',
        'models/model_a_svm_checkpoint.pkl',
        'models/tfidf_vectorizer_checkpoint.pkl',
        'models/distractor_ranker.pkl',
    ] if not os.path.exists(p)
]
if _missing_models:
    st.warning(
        f"**Model files not found** ({len(_missing_models)} missing). "
        "Run `python src/model_a_train_improved.py` and `python src/train_distractor_ranker.py` "
        "to generate all required model checkpoints.",
        icon=""
    )

for key, default in {
    'article': None,
    'question': None,
    'correct_ans': None,
    'options': [],
    'hint_level': 0,
    'answer_revealed': False,
    'check_result': None,
    'model_a_prediction': None,
    'latency': 0.0,
    'source': None,
    'raw_options': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def reset_quiz():
    st.session_state['options'] = []
    st.session_state['hint_level'] = 0
    st.session_state['answer_revealed'] = False
    st.session_state['check_result'] = None

def _extract_qa_from_sentence(sentence: str):

    try:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)

        for i, (word, pos) in enumerate(tagged):
            if word.lower() in ('was', 'is', 'were', 'are') and 0 < i < len(tagged) - 2:
                subj_tokens = tagged[:i]
                subj_words = [w for w, p in subj_tokens
                              if p not in (',', '.', 'CC', 'IN', 'WP', 'WRB')]

                ans_parts = []
                for w, p in tagged[i + 1:]:
                    if w in ('.', '?', '!', ';') or (w == ',' and ans_parts):
                        break
                    ans_parts.append(w)

                ans = " ".join(ans_parts[:6]).strip()
                subj = " ".join(subj_words[-3:]).strip()

                if subj and 1 <= len(ans.split()) <= 6:
                    is_person = any(p in ('NNP', 'NNPS') for _, p in subj_tokens)
                    q_word = "Who" if is_person else "What"
                    return f"{q_word} {word} {subj}?", ans

        for i, (word, pos) in enumerate(tagged):
            if pos in ('VBD', 'VBZ', 'VBP') and 0 < i < len(tagged) - 2:
                subj_tokens = tagged[:i]
                named_entities = [w for w, p in subj_tokens if p in ('NNP', 'NNPS')]
                if not named_entities:
                    continue

                obj_parts = []
                for w, p in tagged[i + 1:]:
                    if w in ('.', '?', '!', ';', ':'):
                        break
                    if p.startswith('NN') or p in ('DT', 'JJ', 'JJR', 'CD', 'PRP$'):
                        obj_parts.append(w)
                    elif obj_parts:
                        break

                ans = " ".join(obj_parts[:5]).strip()
                subject = named_entities[-1]

                if ans and 1 <= len(ans.split()) <= 6:

                    from nltk.stem import WordNetLemmatizer
                    base_verb = WordNetLemmatizer().lemmatize(word.lower(), pos='v')
                    return f"What did {subject} {base_verb}?", ans

        year_match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', sentence)
        if year_match:
            year = year_match.group(1)
            for i, (word, pos) in enumerate(tagged):
                if pos in ('VBD', 'VBZ') and i > 1:
                    subj_before = [w for w, p in tagged[max(0, i-4):i]
                                   if p.startswith('NN') or p.startswith('PRP')]
                    if subj_before:
                        return f"When did {subj_before[-1]} {word.lower()}?", year
    except Exception:
        pass

    return None, None

def generate_question_from_article(article: str):

    sentences = nltk.sent_tokenize(article)
    valid = [s for s in sentences if len(s.split()) > 8 and '?' not in s]
    if not valid:
        valid = [s for s in sentences if len(s.split()) > 5]
    if not valid:
        valid = sentences

    random.shuffle(valid)

    for sent in valid[:12]:
        q, a = _extract_qa_from_sentence(sent)
        if q and a and len(a.strip()) > 1:
            return q, a

    for sent in valid[:3]:
        try:
            tokens = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(tokens)
            nouns = [w for w, p in tagged if p in ('NNP', 'NNPS', 'NN')]
            if nouns:
                topic = nouns[0]
                ans_candidates = [w for w, p in tagged
                                  if p.startswith('NN') and w.lower() != topic.lower()]
                ans = ans_candidates[0] if ans_candidates else nouns[-1]
                return f"According to the passage, what is mentioned about {topic}?", ans
        except Exception:
            continue

    words = valid[0].split()
    return f"What is the passage mainly about?", words[0]

st.markdown("<h1 class='main-title'>RACE AI Quiz Generator</h1>", unsafe_allow_html=True)

page = st.radio(
    "Navigation",
    ["Article Input", "Quiz View", "Hint Panel", "Analytics Dashboard"],
    horizontal=True,
    label_visibility="collapsed"
)

if page == "Article Input":
    st.subheader("Step 1 — Provide a Reading Passage")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        tab_paste, tab_random = st.tabs(["Paste Custom Article", "Load RACE Sample"])

        with tab_paste:
            pasted = st.text_area(
                "Paste your reading passage here:",
                height=280,
                value=st.session_state.get('article', '') if st.session_state['source'] == 'custom' else '',
                placeholder="Enter a reading passage of at least 3–4 sentences...",
                key="paste_area"
            )

        with tab_random:
            st.markdown("Click the button below to load a random passage from the RACE dataset.")
            if st.button(" Load Random RACE Sample", use_container_width=True):
                idx = random.randint(0, len(df) - 1)
                row = df.iloc[idx]
                st.session_state['article'] = row['article']
                st.session_state['question'] = row['question']
                st.session_state['correct_ans'] = str(row[row['answer']])
                st.session_state['raw_options'] = {
                    'A': str(row['A']), 'B': str(row['B']),
                    'C': str(row['C']), 'D': str(row['D']),
                    'answer': row['answer']
                }
                st.session_state['source'] = 'race'
                reset_quiz()
                st.success(f"Loaded RACE sample! The question and all 4 options come from the dataset.")

            if st.session_state['source'] == 'race' and st.session_state['article']:
                st.text_area("Loaded Passage:", st.session_state['article'], height=200, disabled=True)

    with col_right:
        st.markdown("#### Instructions")
        st.info(
            "**Option A:** Paste any article and click Submit — the system will generate a question and options automatically.\n\n"
            "**Option B:** Load a RACE dataset sample which includes a real question and answer choices.\n\n"
            "After submitting, navigate to **Quiz View** to answer the question."
        )

    st.markdown("---")

    if st.button("Submit & Run Inference", type="primary", use_container_width=True):
        article_to_use = pasted.strip() if pasted and pasted.strip() else None

        if article_to_use and len(article_to_use.split()) < 20:
            st.markdown(
                '<div class="error-box">'
                ' Please enter a longer passage (at least 20 words) '
                'for meaningful quiz generation.</div>',
                unsafe_allow_html=True
            )
        elif article_to_use:
            with st.spinner("Step 1/3 — Generating question from article..."):
                start = time.time()
                q, a = generate_question_from_article(article_to_use)

            with st.spinner("Step 2/3 — Running Model B: generating distractors & hints..."):
                distractors = generate_distractors(article_to_use, a, num_distractors=3)
                opts = distractors + [a]
                random.shuffle(opts)

            with st.spinner("Step 3/3 — Running Model A: scoring all 4 options..."):
                pre_prediction = predict_correct_option(article_to_use, q, opts)

            st.session_state['article']          = article_to_use
            st.session_state['question']         = q
            st.session_state['correct_ans']      = a
            st.session_state['source']           = 'custom'
            st.session_state['raw_options']      = None
            st.session_state['latency']          = time.time() - start
            reset_quiz()
            st.session_state['options']          = opts
            st.session_state['model_a_prediction'] = pre_prediction
            st.success("Done! Navigate to **Quiz View** to answer your question.")

        elif st.session_state['source'] == 'race' and st.session_state['article']:
            with st.spinner("Step 1/2 — Running Model B: preparing distractors..."):
                start = time.time()
                raw  = st.session_state['raw_options']
                opts = [raw['A'], raw['B'], raw['C'], raw['D']]
                random.shuffle(opts)

            with st.spinner("Step 2/2 — Running Model A: scoring all 4 options..."):
                pre_prediction = predict_correct_option(
                    st.session_state['article'],
                    st.session_state['question'],
                    opts
                )

            st.session_state['latency'] = time.time() - start
            reset_quiz()
            st.session_state['options']          = opts
            st.session_state['model_a_prediction'] = pre_prediction
            st.success("Done! Navigate to **Quiz View** to answer your question.")
        else:
            st.markdown(
                '<div class="error-box">'
                ' No article loaded. Please paste a reading passage '
                'or load a RACE sample first.</div>',
                unsafe_allow_html=True
            )

elif page == "Quiz View":
    st.subheader("Step 2 — Answer the Question")

    if not st.session_state['article'] or not st.session_state['options']:
        st.warning("No quiz loaded yet. Please go to **Article Input** and click Submit first.")
    else:
        with st.expander(" View Reference Article", expanded=False):
            st.write(st.session_state['article'])

        st.markdown("---")
        st.markdown("#### Question")
        st.info(f"**{st.session_state['question']}**")
        st.markdown(" ")

        opts = st.session_state['options']
        labels = ['A', 'B', 'C', 'D']

        selected = st.radio(
            "Select your answer:",
            opts,
            format_func=lambda x: f"{labels[opts.index(x)]}.  {x}",
            key="quiz_radio"
        )

        if st.button("Check Answer", type="primary"):
            st.session_state['check_result'] = selected

            if st.session_state.get('model_a_prediction') is None:
                with st.spinner("Running Model A inference..."):
                    prediction = predict_correct_option(
                        st.session_state['article'],
                        st.session_state['question'],
                        opts
                    )
                st.session_state['model_a_prediction'] = prediction

        if st.session_state['check_result'] is not None:
            chosen = st.session_state['check_result']
            correct = st.session_state['correct_ans']
            prediction = st.session_state.get('model_a_prediction')

            st.markdown("#### Results")
            for i, opt in enumerate(opts):
                suffix = ""
                if prediction:
                    lr_score = prediction['lr_scores'].get(opt, 0)
                    suffix = f" &nbsp;<span style='font-size:0.8em;color:#6B7280;'>Model A score: {lr_score:.3f}</span>"
                    if opt == prediction['lr_predicted']:
                        suffix += " "

                if opt == correct and opt == chosen:
                    st.markdown(
                        f'<div class="option-box option-correct"> {labels[i]}. {opt} — <strong>Correct! Your answer.</strong>{suffix}</div>',
                        unsafe_allow_html=True
                    )
                elif opt == correct:
                    st.markdown(
                        f'<div class="option-box option-correct"> {labels[i]}. {opt} — Correct answer{suffix}</div>',
                        unsafe_allow_html=True
                    )
                elif opt == chosen:
                    st.markdown(
                        f'<div class="option-box option-wrong"> {labels[i]}. {opt} — Your choice (incorrect){suffix}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="option-box">{labels[i]}. {opt}{suffix}</div>',
                        unsafe_allow_html=True
                    )

            if prediction:
                st.markdown("---")
                ml_col1, ml_col2 = st.columns(2)
                with ml_col1:
                    lr_pick = prediction['lr_predicted']
                    match = " Correct" if lr_pick == correct else " Incorrect"
                    st.info(f"**Model A (Logistic Regression) predicts:** {lr_pick}  \n{match}")
                with ml_col2:
                    svm_pick = prediction['svm_predicted']
                    match_svm = " Correct" if svm_pick == correct else " Incorrect"
                    st.info(f"**Model A (Linear SVM) predicts:** {svm_pick}  \n{match_svm}")

            if chosen != correct:
                st.info(" Tip: Check the **Hint Panel** tab for graduated hints.")

elif page == "Hint Panel":
    st.subheader("Step 3 — Graduated Hints")

    if not st.session_state['article']:
        st.warning("No quiz loaded yet. Please go to **Article Input** first.")
    else:
        st.markdown(
            "Hints are revealed **one at a time**, from general to specific. "
            "**Reveal Answer** only becomes available after all 3 hints have been viewed."
        )
        st.markdown("---")

        hints = get_graduated_hints(
            st.session_state['article'],
            st.session_state['question'],
            st.session_state['correct_ans']
        )

        hint_styles = ["hint-box", "hint-box hint-box-2", "hint-box hint-box-3"]
        hint_labels = [" Hint 1 (General)", " Hint 2 (Specific)", " Hint 3 (Near-Explicit)"]

        level = st.session_state['hint_level']

        for i in range(level):
            hk = f"Hint {i+1}"
            st.markdown(
                f'<div class="{hint_styles[i]}"><strong>{hint_labels[i]}</strong><br>{hints[hk]}</div>',
                unsafe_allow_html=True
            )

        if level < 3:
            btn_label = f"Reveal {'Hint 1' if level == 0 else f'Next Hint ({level+1}/3)'}"
            if st.button(btn_label, use_container_width=True):
                st.session_state['hint_level'] += 1
                st.rerun()
        else:
            st.success("All 3 hints revealed.")
            st.markdown("---")
            if not st.session_state['answer_revealed']:
                if st.button(" Reveal Answer", type="primary", use_container_width=True):
                    st.session_state['answer_revealed'] = True
                    st.rerun()
            else:
                st.markdown(
                    f'<div class="hint-box hint-box-3"><strong> Answer:</strong> {st.session_state["correct_ans"]}</div>',
                    unsafe_allow_html=True
                )

elif page == "Analytics Dashboard":
    st.subheader("Analytics Dashboard")

    latency = st.session_state.get('latency', 0)
    l1, l2 = st.columns(2)
    l1.metric("Last Inference Latency (Model A + B)", f"{latency:.3f} s" if latency > 0 else "Run a quiz first")
    l2.metric("Dataset Size", "87,866 questions (RACE)")

    st.markdown("---")

    st.markdown("#### Model A — Full Metric Comparison Table (Validation Set)")
    comparison_df = pd.DataFrame({
        'Model':         ['Linear SVM', 'Logistic Regression', 'Complement Naive Bayes',
                          'K-Means (Unsupervised)', 'Hard Voting (SVM+LR+NB)', 'Soft Voting (cal.SVM+LR+NB)'],
        'Type':          ['Supervised', 'Supervised', 'Supervised',
                          'Unsupervised', 'Ensemble', 'Ensemble'],
        'Accuracy':      ['58%', '58%', '~54%', 'Purity: 0.82', '58%', '59%'],
        'Precision (1)': ['0.30', '0.30', '0.27', '—', '0.31', '0.31'],
        'Recall (1)':    ['0.54', '0.54', '0.60', '—', '0.53', '0.55'],
        'F1 (weighted)': ['0.60', '0.60', '0.57', 'Silhouette: 0.49', '0.61', '0.62'],
        'MC Accuracy':   ['35.48%', '35.73%', '~30%', '—', '—', '~36%'],
    })
    st.dataframe(comparison_df.set_index('Model'), use_container_width=True)

    st.caption(
        "Binary classification: 1 = correct option, 0 = incorrect. "
        "MC Accuracy = model picks right answer from 4 options (random = 25%). "
        "K-Means evaluated via Silhouette Score and Clustering Purity (no labels used)."
    )

    st.markdown("---")

    st.markdown("#### Ensemble Strategy & Improvement Justification")
    st.markdown("""
    **Hard Voting (≥ 3 classifiers: SVM + LR + NB)**
    - Majority vote across 3 diverse classifiers
    - Reduces individual model variance by aggregating predictions
    - F1 (weighted): **0.61** vs SVM/LR individual: **0.60**  improvement demonstrated

    **Soft Voting (Calibrated SVM + LR + NB)**
    - Averages predicted class probabilities across 3 classifiers
    - SVM calibrated using `CalibratedClassifierCV(cv='prefit')` on validation set
    - F1 (weighted): **0.62** — best overall  outperforms all individual models

    **K-Means (Unsupervised)**
    - Clustering purity: **~0.82** — clusters align strongly with correct/incorrect label
    - Silhouette score: **0.49** — moderate cluster separation
    - Supervised models significantly outperform (0.58 acc vs ~0.50 random for K-Means)
    """)

    st.markdown("---")

    st.markdown("#### Confusion Matrices (Validation Set)")
    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown("**Linear SVM**")
        cm_svm = pd.DataFrame(
            [[15544, 10817], [4059, 4728]],
            index=['Actual: Incorrect', 'Actual: Correct'],
            columns=['Pred: Incorrect', 'Pred: Correct']
        )
        st.dataframe(cm_svm, use_container_width=True)
    with cm2:
        st.markdown("**Logistic Regression**")
        cm_lr = pd.DataFrame(
            [[15552, 10809], [4053, 4734]],
            index=['Actual: Incorrect', 'Actual: Correct'],
            columns=['Pred: Incorrect', 'Pred: Correct']
        )
        st.dataframe(cm_lr, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Model B — NLP Generation Metrics (Distractor Quality)")
    m1, m2, m3 = st.columns(3)
    m1.metric("BLEU Score", "0.1126", help="Bigram overlap with gold distractors")
    m2.metric("ROUGE-L Score", "0.1040", help="Longest common subsequence F1")
    m3.metric("METEOR Score", "0.1776", help="Semantic + synonym matching")

    st.markdown("---")

    st.markdown("#### EDA Visualizations")
    eda_images = [
        ('report/images/answer_dist_and_article_len.png', 'Answer Distribution & Article Length'),
        ('report/images/question_option_lengths.png',     'Question & Option Lengths'),
        ('report/images/outlier_detection.png',           'Outlier Detection (IQR + Z-Score)'),
        ('report/images/correlation_analysis.png',        'Feature Correlation & Relationships'),
        ('report/images/missing_values.png',              'Missing Value Heatmap'),
    ]
    found = [(p, t) for p, t in eda_images if os.path.exists(p)]
    if found:
        for img_path, title in found:
            st.markdown(f"**{title}**")
            st.image(img_path, use_container_width=True)
    else:
        st.warning(
            "EDA images not found. Run the following command to generate them:\n\n"
            "```\npython src/EDA_graphs.py\n```"
        )

    st.markdown("---")

    st.markdown("#### Export All Metrics")
    csv_data = (
        "model,type,accuracy,precision_1,recall_1,f1_weighted,mc_accuracy\n"
        "Linear SVM,Supervised,0.58,0.30,0.54,0.60,0.3548\n"
        "Logistic Regression,Supervised,0.58,0.30,0.54,0.60,0.3573\n"
        "K-Means,Unsupervised,silhouette=0.49,—,—,—,—\n"
        "Hard Voting Ensemble,Ensemble,0.58,0.31,0.53,0.61,—\n"
        ",,,,,,\n"
        "model_b_metric,value,,,,, \n"
        "BLEU,0.1126,,,,,\n"
        "ROUGE-L,0.1040,,,,,\n"
        "METEOR,0.1776,,,,,\n"
    )
    st.download_button(
        label="Download All Metrics as CSV",
        data=csv_data,
        file_name="model_metrics.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("#### Model Checkpoint Files")
    st.code("""
    models/model_a_svm_checkpoint.pkl      — Linear SVM (supervised)
    models/model_a_lr_checkpoint.pkl       — Logistic Regression (supervised)
    models/model_a_kmeans_checkpoint.pkl   — K-Means (unsupervised)
    models/model_a_ensemble_checkpoint.pkl — Hard Voting Ensemble
    models/tfidf_vectorizer_checkpoint.pkl — TF-IDF feature extractor
    models/distractor_ranker.pkl           — Model B ML Ranker
    models/distractor_ohe.pkl              — OHE vectorizer (Model B)
    models/distractor_tfidf.pkl            — TF-IDF vectorizer (Model B)
    """)
    st.caption("System Status: Operational")
