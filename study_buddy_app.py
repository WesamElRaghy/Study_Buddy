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

MODEL_PATH = os.path.join(os.getcwd(), "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please make sure the model is downloaded.")
    st.stop()

if "backend" not in st.session_state:
    with st.spinner("Loading model... This may take a minute or two..."):
        st.session_state.backend = StudyBuddyBackend(MODEL_PATH)
    st.success("Model loaded successfully!")

st.title("📚 Local Study Buddy")
st.markdown("""
This is your personal study assistant that runs completely offline on your computer.
Ask it to solve problems, generate practice questions, or query uploaded PDFs!
""")

with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Response Speed", min_value=0.1, max_value=0.7, value=0.2, step=0.1,
                           help="Lower values (0.1-0.3) produce faster responses.")
    st.warning("⚡ For fastest performance, keep Response Speed at 0.1-0.2")
    st.header("About")
    st.markdown("""
    **Local Study Buddy** is powered by Mistral 7B, running on your CPU.
    No data leaves your computer - everything runs locally!
    """)

tab1, tab2, tab3 = st.tabs(["📝 Problem Solver", "🔄 Practice Question Generator", "📄 PDF Query"])

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
    
    solution_placeholder = st.empty()
    if solve_button and question_input:
        solution_placeholder.markdown("### Solution:\nThinking...")
        result = st.session_state.backend.answer_question(question_input, temperature)
        full_text = result["answer"]
        st.session_state.time_taken = result["time_taken"]
        for i in range(1, len(full_text) + 1, 3):
            solution_placeholder.markdown(f"### Solution:\n{full_text[:i]}")
            time.sleep(0.01)
        solution_placeholder.markdown(f"### Solution:\n{full_text}")

with tab2:
    st.header("Practice Question Generator")
    reference_question = st.text_area("Enter a reference question:", 
                                    height=100, 
                                    placeholder="Example: What is the derivative of f(x) = x³ + 2x² - 5x + 3?")
    num_questions = st.slider("Number of questions to generate", min_value=1, max_value=5, value=3)
    generate_button = st.button("Generate Practice Questions", type="primary")
    questions_placeholder = st.empty()
    if generate_button and reference_question:
        questions_placeholder.markdown("### Similar Practice Questions:\nThinking...")
        practice_questions = st.session_state.backend.generate_practice_questions(
            reference_question, num_questions, temperature
        )
        for i in range(1, len(practice_questions) + 1, 3):
            questions_placeholder.markdown(f"### Similar Practice Questions:\n{practice_questions[:i]}")
            time.sleep(0.01)
        questions_placeholder.markdown(f"### Similar Practice Questions:\n{practice_questions}")

with tab3:
    st.header("PDF Query")
    st.markdown("Upload a PDF textbook or document and ask questions about its content.")
    
    # Initialize session state for PDF
    if "pdf_data" not in st.session_state:
        st.session_state.pdf_data = None
        st.session_state.pdf_processed = False

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_file:
        pdf_data = uploaded_file.read()
        # Only process if new PDF is different from the current one
        if st.session_state.pdf_data != pdf_data:
            st.session_state.pdf_data = pdf_data
            st.session_state.pdf_processed = False
            with st.spinner("Processing PDF... This may take a moment..."):
                st.session_state.backend.reset_pdf()  # Clear previous PDF
                success = st.session_state.backend.process_pdf(pdf_data)
            if success:
                st.session_state.pdf_processed = True
                st.success("PDF processed successfully!")
            else:
                st.session_state.pdf_processed = False
                st.error("Failed to process PDF. Please try again.")

    # Add Reset PDF button
    if st.session_state.pdf_processed:
        if st.button("Reset PDF", type="secondary"):
            st.session_state.backend.reset_pdf()
            st.session_state.pdf_data = None
            st.session_state.pdf_processed = False
            st.success("PDF reset successfully. Upload a new PDF to continue.")

    pdf_question = st.text_area("Ask a question about the PDF:", 
                               height=100, 
                               placeholder="Example: What is erosion according to the textbook?")
    col1, col2 = st.columns([1, 3])
    with col1:
        query_button = st.button("Query PDF", type="primary", use_container_width=True)
    with col2:
        if "pdf_time_taken" in st.session_state:
            st.info(f"Response generated in {st.session_state.pdf_time_taken} seconds")
    
    pdf_answer_placeholder = st.empty()
    if query_button and pdf_question:
        if not st.session_state.pdf_processed:
            pdf_answer_placeholder.error("Please upload and process a PDF first.")
        else:
            pdf_answer_placeholder.markdown("### Answer:\nThinking...")
            result = st.session_state.backend.answer_pdf_question(pdf_question, temperature)
            full_text = result["answer"]
            st.session_state.pdf_time_taken = result["time_taken"]
            for i in range(1, len(full_text) + 1, 3):
                pdf_answer_placeholder.markdown(f"### Answer:\n{full_text[:i]}")
                time.sleep(0.01)
            pdf_answer_placeholder.markdown(f"### Answer:\n{full_text}")
