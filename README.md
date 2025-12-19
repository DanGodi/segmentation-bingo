# Segmentation Bingo 🛰️

Generate and play a custom Bingo game using custom imagery and AI-powered segmentation.

This project allows you to turn a collection of images into a fully playable Bingo game. It uses the **Segment Anything Model (SAM)** to detect features (like pools, cars, boats) and generates statistically "fair" Bingo cards based on the frequency of these features.

## 🌟 Features

*   **Interactive Labeling**: Easy-to-use widget to tag features in your images.
*   **AI Segmentation**: Uses Meta's SAM3 to detect objects and calculate coverage/counts.
*   **Fair Card Generation**: Uses Monte Carlo simulations to ensure all Bingo cards have similar difficulty (expected turns to win).
*   **Printable Assets**: Generates PDF/Image files for the Bingo cards.
*   **Game Presentation**: Creates a slide deck (PDF) to "call" the game by revealing images one by one.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DanGodi/segmentation-bingo.git
    cd satellite-bingo
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You will need a GPU-enabled environment for reasonable performance with the Segment Anything Model.*

## 🔑 Model Access (Important!)

This project uses **SAM 3 (Segment Anything Model 3)** via the `samgeo` library. To use it, you must:

1.  **Request Access**: Go to the [SAM 3 Hugging Face page](https://huggingface.co/facebook/sam3) (or the specific model used) and accept the license terms.
    *   *Note: Ensure you have access to the model checkpoints required by `samgeo`.*
2.  **Get a Token**: Create a [Hugging Face Access Token](https://huggingface.co/settings/tokens) (Read permissions are sufficient).
3.  **Configure Notebook**: In `run_full_game.ipynb`, look for **Step 3** and paste your token into the `HF_TOKEN` variable:
    ```python
    HF_TOKEN = "hf_..." 
    ```

## 🎮 How to Create Your Game

The entire pipeline is orchestrated from a single Jupyter Notebook.

1.  **Prepare Images**: Place your raw images (JPG, PNG, TIF) in a directory of your choice in the main directory and input folder path into the master notebook when prompted.
2.  **Open the Master Notebook**:
    Open `run_full_game.ipynb`.
3.  **Run the Steps**:
    *   **Step 1**: Process images (resizes them to `converted_sat_images/`).
    *   **Step 2**: Run the interactive labeler to select which features (e.g., "Pool", "Ship") are in which image.
    *   **Step 3**: Run the AI analysis. This will generate masks and stats in `mask/`.
    *   **Step 4**: Generate Bingo Cards. You can configure the difficulty and number of cards.
    *   **Step 5**: Generate printable cards in `printable_cards/`.
    *   **Step 6**: Generate the game presentation PDF (`bingo_game_presentation.pdf`).

## 📂 Project Structure

*   `run_full_game.ipynb`: The main entry point for the project.
*   `utils/`: Python modules for each step of the pipeline.
    *   `process_images.py`: Image preprocessing.
    *   `label_images.py`: Interactive labeling widget.
    *   `analyze_segment.py`: SAM inference and stats calculation.
    *   `create_cards.py`: Statistical card generation.
    *   `generate_printable_cards.py`: Card rendering.
    *   `generate_game_pdf.py`: Game presentation generation.
*   `sat_images/`: Input folder for your raw images.
*   `converted_sat_images/`: Processed images used for the game.
*   `mask/`: Output folder for segmentation masks and statistics.
*   `printable_cards/`: Generated Bingo cards ready for printing.

