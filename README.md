# 📌 Context-Aware Multi-Aspect Sentiment Analysis (CAM-ABSA)

This repository contains an implementation of **Context-Aware Multi-Aspect Sentiment Analysis (CAM-ABSA)** in **Python**.  
The goal is to extract **multiple aspects** from a review (ex: *battery, camera, display*) and predict the **sentiment polarity** (*positive / neutral / negative*) for each aspect using a **hybrid context-aware approach**.

In addition, this project extracts the **reason/cause** behind the predicted sentiment using a **Question Answering (QA) model**.

---

## 🚀 Key Features

- ✅ Multi-aspect extraction from a single review using **spaCy**
- ✅ Context-based snippet generation using clause boundaries + local windowing
- ✅ Sentiment prediction using:
  - **VADER Sentiment** (rule-based scoring)
  - **RoBERTa Transformer** (deep learning)
- ✅ Hybrid late-fusion strategy (**Transformer + VADER**) using **α (alpha)**
- ✅ Contrast handling for statements like: **but, however, although, though, yet**
- ✅ Cause extraction using **QA model**
- ✅ Works on Amazon product reviews dataset downloaded via **KaggleHub**
- ✅ Output saved in **JSON + CSV**
- ✅ Visualization using **Plotly + Matplotlib**
- ✅ Interactive testing via **Gradio Web UI**

---

## 🧠 Models Used

### Sentiment Classification (Transformer)
- `siebert/sentiment-roberta-large-english`  
(Binary POS/NEG mapped into a 3-way distribution)

### Cause/Reason Extraction (Question Answering)
- `deepset/roberta-base-squad2`

---

## 🛠️ Tech Stack

- Python
- Transformers (HuggingFace)
- NLTK (VADER)
- Pandas, NumPy
- KaggleHub
- Gradio

---

## 📂 Dataset

Dataset is downloaded automatically using KaggleHub:

- Kaggle Dataset: `arhamrumi/amazon-product-reviews`

The script automatically selects the first `.csv` file available in the dataset folder and detects review text columns such as:
`reviewText`, `review`, `text`, `content`, etc.

---

## ⚙️ Installation

### 1) Install dependencies
```bash
pip install transformers==4.44.2 torch torchvision torchaudio accelerate nltk spacy pandas tqdm plotly==5.24.1 kagglehub gradio==4.44.1
```

### 2) Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 3) Download VADER lexicon
```python
import nltk
nltk.download("vader_lexicon")
```

---

## ▶️ Run the Project

Run the script (or notebook) and it will:

✅ Download dataset  
✅ Process first `5000` reviews (configurable)  
✅ Extract aspects + sentiment + cause span  
✅ Save JSON/CSV output files  
✅ Show plots  
✅ Launch Gradio UI  

---

## 🧪 CAM-ABSA Pipeline (High Level)

1. Load dataset (KaggleHub)
2. Clean & clip very long reviews
3. Extract aspect candidates (noun chunks + nouns)
4. Create a local context snippet around each aspect
5. Predict sentiment:
   - VADER prior distribution (3-way)
   - Transformer probabilities (binary → 3-way)
6. Fuse with late fusion:  
   `p_fused = (1 - α) * p_transformer + α * p_vader`
7. Apply contrast cue weighting (after "but"/"however")
8. Extract cause span using QA model
9. Save results + generate visualizations

---

## 📊 Outputs Generated

All outputs are stored in:

```
outputs/
```

### ✅ Files
- `cam_absa_5000.json` → Full structured output for each review
- `cam_absa_5000_aspects.csv` → Aspect-level sentiment predictions
- `cam_absa_500_review_summaries.csv` → Top-N aspects summary per review
- `model_accuracy_comparison.png` → Accuracy comparison chart

---

## 📌 Example Output

Input Review:
```
The phone was great but the battery life was not good.
```

Output (sample):
```
• Aspect: phone
  Sentiment: positive (conf 0.90)

• Aspect: battery life
  Sentiment: negative (conf 0.86)
  Cause: not good
```

---

## 🌐 Gradio Interface

The project includes a Gradio web app where you can:

- Enter a custom review
- Provide manual aspects (optional)
- Adjust fusion alpha (α)
- Change the local context window size

---

## 📈 Visualizations

The code generates:
- Aspect sentiment distribution (bar chart)
- Top extracted aspects frequency
- Rating vs positive probability correlation (if rating column exists)
- Confusion matrix + classification report (weak labels)
- Accuracy comparison bar chart

---

## 🏷️ Weak Label Evaluation (Optional)

If the dataset contains star ratings (`overall`, `rating`, `stars`, etc.), the project maps:
- ⭐⭐⭐⭐/⭐⭐⭐⭐⭐ → Positive
- ⭐/⭐⭐ → Negative
- ⭐⭐⭐ → Neutral

Then it evaluates predictions using:
✅ Confusion Matrix  
✅ Approx Accuracy  
✅ Classification Report  

---

## ✨ Future Improvements

- Replace naive aspect extraction with ABSA taggers (BERT-CRF)
- Fine-tune transformer for 3-class sentiment
- Improve neutral sentiment modeling
- Optimize runtime for full dataset processing

---

## 👨‍💻 Authors

Developed as a project on **Context-Aware Multi-Aspect Sentiment Analysis (CAM-ABSA)** using hybrid NLP + deep learning techniques.

