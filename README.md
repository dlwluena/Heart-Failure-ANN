# Heart Failure Prediction — Artificial Neural Network (ANN)

This project uses a **PyTorch-based Artificial Neural Network (ANN)** to predict the likelihood of **heart failure** based on patient medical records.  
It demonstrates a practical machine learning workflow: data preprocessing, model training, evaluation, and visualization.

---

## Model Architecture

- **Framework:** PyTorch  
- **Model Type:** Feedforward Artificial Neural Network (ANN)  
- **Layers:**
  - Input layer (number of features from dataset)
  - Hidden layers with ReLU activation
  - Output layer with Sigmoid activation (binary output)
- **Loss Function:** Binary Cross Entropy (BCELoss)  
- **Optimizer:** Adam  

---

## Dataset

**Source:** [Kaggle — Heart Failure Clinical Records Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

**Features include:**
- Age, Sex, Anaemia, Diabetes, High Blood Pressure  
- Ejection Fraction, Serum Creatinine, Serum Sodium  
- Time, Platelets, Smoking, etc.  

**Target:** `DEATH_EVENT` (1 = died, 0 = survived)

---

## Workflow

1. Data preprocessing and normalization  
2. Splitting data into training and test sets  
3. Defining and training the ANN model in PyTorch  
4. Evaluating performance metrics (Accuracy, ROC-AUC, Confusion Matrix)  
5. Visualizing ROC curve and metrics  

---

## Example Metrics

| Metric | Value (Example) |
|:--|:--|
| Accuracy | 0.87 |
| ROC-AUC  | 0.91 |

> *These values depend on random initialization and training splits.*

---

## Requirements

Create a virtual environment and install dependencies:

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib
```

## How to Run
jupyter notebook Heart_Failure.ipynb

## 📁 File Structure

```text
Heart_Failure_ANN/
│
├── Heart_Failure.ipynb     # Main Jupyter Notebook (model training & evaluation)
├── README.md               # Project documentation (this file)
└── requirements.txt        # Python dependencies (optional)
```

## Note!!
During training, you may see the following message:
"UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate."

This is not an error — just a PyTorch update notice.
It appears because verbose in ReduceLROnPlateau is now deprecated.
You can safely ignore it or remove the verbose argument from the scheduler.
