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

# Initialize chat history in session state
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Chat 1": {
            "problem_solver": [],  # List of {"question": str, "answer": str, "time_taken": float}
            "practice_questions": [],  # List of {"question": str, "answer": str}
            "pdf_query": []  # List of {"question": str, "answer": str, "time_taken": float}
        }
    }
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1

st.title("📚 Local Study Buddy")
st.markdown("""
This is your personal study assistant that runs completely offline on your computer.
Ask it to solve problems, generate practice questions, or query uploaded PDFs!
""")

# Sidebar for settings and chat management
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Response Speed", min_value=0.1, max_value=0.7, value=0.2, step=0.1,
                           help="Lower values (0.1-0.3) produce faster responses.")
    st.warning("⚡ For fastest performance, keep Response Speed at 0.1-0.2")
    
    st.header("Chat Management")
    chat_name = st.selectbox("Select Chat", options=list(st.session_state.chats.keys()), index=list(st.session_state.chats.keys()).index(st.session_state.current_chat))
    if chat_name != st.session_state.current_chat:
        st.session_state.current_chat = chat_name
    
    if st.button("New Chat", type="primary"):
        st.session_state.chat_counter += 1
        new_chat_name = f"Chat {st.session_state.chat_counter}"
        st.session_state.chats[new_chat_name] = {
            "problem_solver": [],
            "practice_questions": [],
            "pdf_query": []
        }
        st.session_state.current_chat = new_chat_name
        st.session_state.pdf_data = None
        st.session_state.pdf_processed = False
        st.session_state.backend.reset_pdf()
        st.success(f"Started {new_chat_name}")
    
    st.header("About")
    st.markdown("""
    **Local Study Buddy** is powered by Mistral 7B, running on your CPU.
    No data leaves your computer - everything runs locally!
    """)

tab1, tab2, tab3 = st.tabs(["📝 Problem Solver", "🔄 Practice Question Generator", "📄 PDF Query"])

with tab1:
    st.header("Problem Solver")
    
    # Display chat history for Problem Solver
    st.subheader("Chat History")
    for entry in st.session_state.chats[st.session_state.current_chat]["problem_solver"]:
        with st.expander(f"Question: {entry['question'][:50]}..."):
            st.markdown(f"**Question**: {entry['question']}")
            st.markdown(f"**Answer**: {entry['answer']}")
            st.markdown(f"**Time Taken**: {entry['time_taken']} seconds")
    
    question_input = st.text_area("Enter your study question or problem:", 
                                 height=100, 
                                 placeholder="Example: What is the derivative of f(x) = x³ + 2x² - 5x + 3?",
                                 key=f"problem_solver_input_{st.session_state.current_chat}")
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
        # Save to chat history
        st.session_state.chats[st.session_state.current_chat]["problem_solver"].append({
            "question": question_input,
            "answer": full_text,
            "time_taken": result["time_taken"]
        })

with tab2:
    st.header("Practice Question Generator")
    
    # Display chat history for Practice Question Generator
    st.subheader("Chat History")
    for entry in st.session_state.chats[st.session_state.current_chat]["practice_questions"]:
        with st.expander(f"Reference Question: {entry['question'][:50]}..."):
            st.markdown(f"**Reference Question**: {entry['question']}")
            st.markdown(f"**Generated Questions**: {entry['answer']}")
    
    reference_question = st.text_area("Enter a reference question:", 
                                    height=100, 
                                    placeholder="Example: What is the derivative of f(x) = x³ + 2x² - 5x + 3?",
                                    key=f"practice_questions_input_{st.session_state.current_chat}")
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
        # Save to chat history
        st.session_state.chats[st.session_state.current_chat]["practice_questions"].append({
            "question": reference_question,
            "answer": practice_questions
        })

with tab3:
    st.header("PDF Query")
    st.markdown("Upload a PDF textbook or document and ask questions about its content.")
    
    # Initialize session state for PDF
    if "pdf_data" not in st.session_state:
        st.session_state.pdf_data = None
        st.session_state.pdf_processed = False

    # Display chat history for PDF Query
    st.subheader("Chat History")
    for entry in st.session_state.chats[st.session_state.current_chat]["pdf_query"]:
        with st.expander(f"Question: {entry['question'][:50]}..."):
            st.markdown(f"**Question**: {entry['question']}")
            st.markdown(f"**Answer**: {entry['answer']}")
            st.markdown(f"**Time Taken**: {entry['time_taken']} seconds")
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key=f"pdf_uploader_{st.session_state.current_chat}")
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
                               placeholder="Example: What is erosion according to the textbook?",
                               key=f"pdf_query_input_{st.session_state.current_chat}")
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
            # Save to chat history
            st.session_state.chats[st.session_state.current_chat]["pdf_query"].append({
                "question": pdf_question,
                "answer": full_text,
                "time_taken": result["time_taken"]
            })
