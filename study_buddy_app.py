import streamlit as st
import os
from ct_backend import StudyBuddyBackend

st.set_page_config(
    page_title="Local Study Buddy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path to your downloaded model
MODEL_PATH = os.path.join(os.getcwd(), "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please make sure the model is downloaded.")
    st.stop()

# Initialize session state for storing the backend
if "backend" not in st.session_state:
    with st.spinner("Loading model... This may take a minute or two..."):
        st.session_state.backend = StudyBuddyBackend(MODEL_PATH)
    st.success("Model loaded successfully!")

# App title and description
st.title("📚 Local Study Buddy")
st.markdown("""
This is your personal study assistant that runs completely offline on your computer.
Ask it to solve problems step-by-step or generate similar practice questions!
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                          help="Higher values make responses more creative, lower values make them more deterministic")
    
    st.header("About")
    st.markdown("""
    **Local Study Buddy** is powered by Mistral 7B, running entirely on your CPU.
    No data leaves your computer - everything runs locally!
    """)

# Tab-based interface
tab1, tab2 = st.tabs(["📝 Problem Solver", "🔄 Practice Question Generator"])

with tab1:
    st.header("Problem Solver")
    
    question_input = st.text_area("Enter your study question or problem:", 
                                height=100, 
                                placeholder="Example: What is the derivative of f(x) = x³ + 2x² - 5x + 3?")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        solve_button = st.button("Solve Step-by-Step", type="primary", use_container_width=True)
    
    with col2:
        if "time_taken" in st.session_state:
            st.info(f"Response generated in {st.session_state.time_taken} seconds")
    
    if solve_button and question_input:
        with st.spinner("Thinking..."):
            result = st.session_state.backend.answer_question(question_input, temperature)
            st.session_state.time_taken = result["time_taken"]
        
        st.markdown("### Solution:")
        st.markdown(result["answer"])
        
with tab2:
    st.header("Practice Question Generator")
    
    reference_question = st.text_area("Enter a reference question:", 
                                     height=100, 
                                     placeholder="Example: What is the derivative of f(x) = x³ + 2x² - 5x + 3?")
    
    num_questions = st.slider("Number of questions to generate", min_value=1, max_value=5, value=3)
    
    generate_button = st.button("Generate Practice Questions", type="primary")
    
    if generate_button and reference_question:
        with st.spinner("Generating similar questions..."):
            practice_questions = st.session_state.backend.generate_practice_questions(
                reference_question, num_questions, temperature
            )
        
        st.markdown("### Similar Practice Questions:")
        st.markdown(practice_questions)
