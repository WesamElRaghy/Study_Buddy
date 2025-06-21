# Study Buddy

A local, offline study assistant powered by Mistral 7B that runs entirely on your CPU.

## Features

- Solve problems step-by-step
- Generate similar practice questions
- Beautiful Streamlit interface
- Completely local and private - no data leaves your computer

## Requirements

- Ubuntu (or other Linux distribution)
- Python 3.8+
- ~8GB RAM minimum (16GB recommended)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/study-buddy.git
   cd study-buddy
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download the model:
   ```bash
   mkdir -p models
   # Download Mistral 7B Instruct v0.2 (4GB file)
   python download_model.py
   ```
4. Run the application:
   ```bash
   streamlit run study_buddy_app.py
   ```


## Usage

1. Enter a study question in the "Problem Solver" tab
2. Click "Solve Step-by-Step" to get a detailed solution
3. Or switch to the "Practice Question Generator" tab to create similar practice problems


## Note on Model Files

The model file is not included in this repository due to its large size (~4GB).
Use the download_model.py script to download it, or manually download from:
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
