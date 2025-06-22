import streamlit as st
import os
import time
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
    
    # Lower default temperature for faster responses
    temperature = st.slider("Response Speed", min_value=0.1, max_value=0.7, value=0.2, step=0.1,
                          help="Lower values (0.1-0.3) produce much faster responses.")
    
    # Add a performance tip
    st.warning("⚡ For fastest performance, keep the Response Speed at 0.1-0.2")
    
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
    
    # Create a placeholder for the solution
    solution_placeholder = st.empty()
    
    if solve_button and question_input:
        # Show thinking message
        solution_placeholder.markdown("### Solution:\nThinking...")
        
        # Get the full response
        result = st.session_state.backend.answer_question(question_input, temperature)
        full_text = result["answer"]
        st.session_state.time_taken = result["time_taken"]
        
        # Display the answer incrementally
        for i in range(1, len(full_text) + 1, 3):  # Display 3 chars at a time for speed
            solution_placeholder.markdown(f"### Solution:\n{full_text[:i]}")
            time.sleep(0.01)  # Small delay for visual effect
        
        # Final complete display
        solution_placeholder.markdown(f"### Solution:\n{full_text}")
        
with tab2:
    st.header("Practice Question Generator")
    
    reference_question = st.text_area("Enter a reference question:", 
                                     height=100, 
                                     placeholder="Example: What is the derivative of f(x) = x³ + 2x² - 5x + 3?")
    
    num_questions = st.slider("Number of questions to generate", min_value=1, max_value=5, value=3)
    
    generate_button = st.button("Generate Practice Questions", type="primary")
    
    # Create a placeholder for the questions
    questions_placeholder = st.empty()
    
    if generate_button and reference_question:
        # Show thinking message
        questions_placeholder.markdown("### Similar Practice Questions:\nThinking...")
        
        # Get the practice questions
        practice_questions = st.session_state.backend.generate_practice_questions(
            reference_question, num_questions, temperature
        )
        
        # Display incrementally
        for i in range(1, len(practice_questions) + 1, 3):  # Display 3 chars at a time
            questions_placeholder.markdown(f"### Similar Practice Questions:\n{practice_questions[:i]}")
            time.sleep(0.01)  # Small delay for visual effect
        
        # Final complete display
        questions_placeholder.markdown(f"### Similar Practice Questions:\n{practice_questions}")
