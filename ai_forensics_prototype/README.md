# AI Forensics Prototype

This project is a minimal working prototype for analyzing suspicious cybercrime-related text and estimating whether it is more likely AI-generated or human-written.

The app is built for student demos and presentations. It reads a locally downloaded Kaggle phishing email dataset from the project directory, extracts simple forensic-style text features, trains a lightweight Logistic Regression model, and presents the result in a clean Streamlit interface.

## What the Prototype Does

- Accepts suspicious text input such as phishing emails, scam messages, or suspicious chat content
- Detects a phishing dataset CSV from the local project folder or a zip archive in that folder
- Cleans and samples the dataset for fast local demo training
- Extracts forensic-style linguistic features
- Predicts whether the text is more likely AI-generated or human-written
- Displays a confidence score
- Shows the extracted feature values and a short interpretation

## Project Structure

```text
ai_forensics_prototype/
|-- app.py
|-- feature_extraction.py
|-- model.py
|-- sample_data.py
|-- requirements.txt
`-- README.md
```

## Installation

1. Open a terminal in the project directory.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the App

From inside the `ai_forensics_prototype` folder, run:

```powershell
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually `http://localhost:8501`.

## Dataset Usage

- Place the Kaggle phishing dataset file inside `ai_forensics_prototype/`.
- The project can read either a plain `.csv` file or a `.zip` archive containing one or more CSV files.
- The loader automatically looks for common text columns such as `text`, `body`, `content`, or `email`.
- The loader automatically looks for common label columns such as `label`, `class`, or `type`.
- Labels are normalized into binary values where phishing is `1` and legitimate or safe email is `0`.
- For faster demo performance, the training pipeline uses a cleaned sample of the dataset instead of loading every row into the model.

## Notes

- The classifier is intentionally lightweight and meant for demonstration only.
- The dataset loader drops missing rows, removes very short messages, and samples a manageable subset for local use.
- The prediction should be treated as a prototype signal, not a forensic conclusion.
