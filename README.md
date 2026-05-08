# milk_adulteration

Milk Adulteration Detector
A Machine Learning web application that predicts whether a milk sample is Pure, At Risk, or Adulterated — based on its physical and chemical properties.
Built using Python, Scikit-learn, and Streamlit.

# Problem Statement
In India, food adulteration — especially in milk — is a serious public health issue. Adulterants like detergent, water, starch, and urea are commonly mixed into milk, causing long-term health damage especially in children.
Traditional lab testing takes 2-3 days per sample. By the time results arrive, the same milk has already been consumed by hundreds of families.
This project provides an instant, ML-powered decision support system that dairy labs and food inspectors can use on top of their existing infrastructure — no new devices, no extra training required.

# Solution
A trained Random Forest model that takes 7 measurable properties of a milk sample and instantly predicts its quality grade:
GradeMeaning✅ Pure (High)Safe to consume⚠️ At Risk (Medium)Borderline — lab verification recommended❌ Adulterated (Low)Not safe to consume

App Preview

User enters milk sample values → clicks Analyse → gets instant result with confidence score


# Dataset

Source: Kaggle — Milk Quality Prediction Dataset
Size: 1059 samples, 8 columns

# Features used:
PH                  ---   Acidity level of milk

Temperature         ---   Temperature at testing (°C)

Taste               ---  Good (1) / Bad (0)

Odor                ---   Normal (0) / Abnormal (1)

Fat                 --- Present (1) / Absent (0)

Turbidity           ---  Clear (0) / Cloudy (1)

Colour Visual       ---  colour value (240–255)

# Models Trained & Compared
Logistic Regression  ---  Baseline classifier

Decision Tree        ---  Interpretable tree-based model

Random Forest        ---  Best accuracy — selected as final model

SVM                  ---  Support Vector Machine

Random Forest was selected as the final model based on highest accuracy and best confusion matrix results.


# Tech Stack

Language: Python 3.11
ML Library: Scikit-learn
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Web App: Streamlit
Model Saving: Joblib


# How to Run Locally
1. Clone the repository
bashgit clone https://github.com/Kankshipatle/MilkAdulterationDetector.git
cd MilkAdulterationDetector
2. Create virtual environment
bashpython -m venv milk_env

# Windows
milk_env\Scripts\activate

# Mac/Linux
source milk_env/bin/activate
3. Install dependencies
bashpip install -r requirements.txt
4. Run the app
bashstreamlit run app.py
5. Open in browser
http://localhost:8501

# Project Structure
MilkAdulterationDetector/
│
├── app.py                  # Streamlit web application
├── milk_model.pkl          # Trained Random Forest model
├── milk_scaler.pkl         # Fitted StandardScaler
├── milk_notebook.ipynb     # Full ML notebook (EDA + Training)
├── milknew.csv             # Dataset
├── requirements.txt        # Python dependencies
└── README.md               # You are here!

# Project Workflow
1. Load Dataset (1059 milk samples)
3. Exploratory Data Analysis (EDA)
5. Preprocess — encode Grade, scale features
        
6. Train 4 ML models
        
7. Compare accuracy + confusion matrix
        
8. Save best model (Random Forest)
        
9. Build Streamlit web app
        
10. User inputs values → Instant prediction

Real World Impact
This tool is designed as a Decision Support System for:

Government food inspection labs
Dairy processing plants
FSSAI quality control teams

These facilities already measure pH, fat, turbidity daily. This model simply automates the decision-making on top of their existing setup — making adulteration detection 10x faster at zero additional cost.

# Author
Kankshi Patle

GitHub: https://github.com/Kankshipatle
LinkedIn: https://www.linkedin.com/in/kankshi-patle-a03a99316/


License :
This project is open source and available under the MIT License.
