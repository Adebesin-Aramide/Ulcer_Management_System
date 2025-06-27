# 🩺 UlcerMate: AI-Powered Ulcer Management System

UlcerMate is a comprehensive AI-powered solution designed to support individuals living with peptic ulcers by providing:

- A **flare prediction model** to identify potential risks through flare-ups
- A **meal recommendation system** to suggest safer dietary choices based on observed triggers
- An **interactive chatbot** for ulcer-related questions

---

## 🌟 Project Overview

Peptic ulcer disease remains a significant health concern. Many patients lack personalized tools to track triggers, predict flare-ups, or access reliable guidance for daily management. UlcerMate addresses these challenges by integrating multiple AI components into a single, accessible application.

---

## 🤖 1. Chatbot Assistant

### **Description**

The UlcerMate Chatbot leverages a Retrieval-Augmented Generation (RAG) approach to provide **accurate, context-specific responses** to user questions about ulcers.

- **Backend:** LangChain FAISS retriever with a Mistral-7B model deployed via Hugging Face Inference API
- **Dataset:** Curated knowledge base covering ulcer causes, symptoms, treatments, and daily care tips
- **Deployment:** Built using Streamlit and hosted on Streamlit Cloud for easy user access

### **Key Features**

- Retrieves top relevant context from indexed ulcer knowledge base
- Generates clear, concise, medically grounded answers
- Provides disclaimers when uncertain to encourage professional consultation

---

## 🔬 2. Flare-Up Prediction Model

### **Objective**

Predict daily ulcer flare-ups using a combination of patient-reported logs and clinically relevant features.

### **Data Collection & Augmentation**

- Sourced from daily logs over a two-week period, including meals, symptoms, pain scores, stress levels, NSAID usage, and clinical history
- Synthetic augmentation introduced controlled variability while preserving real-world relationships

### **Feature Engineering**

- Multi-label binarization for meals and symptoms
- Label encoding for categorical variables
- Rule-based target creation (`IsFlare`) based on pain scores, symptoms, stress, NSAID use, skipped meals, and known trigger foods
- 5% random noise added for realistic clinical uncertainty

### **Model Training**

- **Algorithm:** Random Forest Classifier
- **Train/Val/Test Split:** 60/20/20 with stratified sampling
- **Evaluation Metrics:** F1 Score and ROC-AUC to balance precision-recall performance under class imbalance

### **Results**

| Dataset       | F1 Score | ROC-AUC | Accuracy |
|---------------|----------|---------|----------|
| Validation    | 0.970    | 0.978   | 97%      |
| Test          | 0.968    | 0.969   | 97%      |

The model achieved **consistent and robust performance**, demonstrating generalizability across datasets.

---

## 🍲 3. Meal Recommendation System

### **Objective**

Recommend safer meals for ulcer patients based on patterns observed in the dataset, aiming to minimize potential triggers.

### **Approach**

- Uses the same feature-engineered dataset as flare prediction.
- Filters meals associated with **lower or no flare-up occurrences** in historical data.
- Suggests these meals as safer options for the user.

### Getting Started

1. Clone the repository

`git clone https://github.com/Adebesin-Aramide/Ulcer_Management_System.git`

`cd Ulcer_Management_System`

2. Install dependencies

`pip install -r requirements.txt`

3. Run the Streamlit app locally:

`streamlit run app.py`

### 🤝 Contributing

Contributions are welcome to expand the knowledge base, improve model performance, and integrate additional ulcer management features.
If you would also like to volunteer to log your meal daily to help curate a larger dataset, you contribute towards it here: 
`https://ulcerapp-db.streamlit.app`


### 🙏 Acknowledgements

Special thanks to the ulcer patients whose data helped shaped this project, and to the open-source community for tools enabling accessible health AI.

### ✨ Connect
For questions, suggestions, or collaboration opportunities, feel free to send me an email: adebesinaramide@gmail.com
