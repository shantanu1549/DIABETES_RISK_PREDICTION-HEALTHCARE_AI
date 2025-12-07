# Healthcare AI: Diabetes Risk Prediction using Machine Learning

## Project Overview
This project builds an end-to-end **Healthcare AI pipeline** to predict the risk of diabetes using patient health records. Multiple machine learning models are trained and evaluated, with a strong emphasis on **model interpretability using Explainable AI (SHAP)**.

The project demonstrates the complete data science workflow — from data preprocessing to model explainability — making it suitable for **academic portfolios and graduate admissions in Data Science / AI**.

---

## Objective
To predict whether a patient is likely to have diabetes (Outcome = 1) based on clinical features such as glucose level, BMI, age, blood pressure, and insulin level.

---

## Dataset
* **Name:** Pima Indians Diabetes Dataset
* **Source:** Kaggle / UCI Machine Learning Repository
* **Target Variable:** `Outcome` (0 = No Diabetes, 1 = Diabetes)

### Features
| Feature | Description |
| :--- | :--- |
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (weight in $\text{kg} / (\text{height in } \text{m})^2$) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age (years) |

---

## Tech Stack
* **Programming Language:** Python
* **Notebook Environment:** Google Colab / Jupyter
* **Libraries Used:**
    * **pandas, numpy** — data processing
    * **matplotlib, seaborn** — visualization
    * **scikit-learn** — preprocessing, modeling, evaluation
    * **xgboost** — gradient boosting classifier (core model)
    * **shap** — Explainable AI (SHAP values)
    * **joblib** — model persistence

---

## Project Workflow

1.  **Data Loading & Cleaning**
    * Loaded dataset and inspected structure.
    * Replaced clinically invalid zero values (in features like Glucose, BMI, BloodPressure) with **median imputation**.

2.  **Exploratory Data Analysis (EDA)**
    * Correlation heatmap to identify feature relationships.
    * Feature distributions and outcome imbalance analysis.

3.  **Preprocessing**
    * Feature scaling using `StandardScaler`.
    * **Stratified train-test split** (80–20) to maintain class balance in both sets.

4.  **Modeling**
    * **Logistic Regression** (baseline)
    * **Random Forest Classifier**
    * **XGBoost Classifier** (advanced ensemble model)

5.  **Evaluation**
    * Metrics: Accuracy, **ROC-AUC**, Confusion Matrix, Classification Report.
    * Visual comparison of ROC Curves.

6.  **Explainable AI (XAI)**
    * Native feature importance (RF & XGBoost).
    * **SHAP (SHapley Additive exPlanations)** summary plots for **global interpretability**.
    * SHAP force plots for understanding **individual predictions**.

7.  **Model Saving**
    * Best-performing model (**XGBoost**) saved using `joblib` for future deployment.

---

## Results

| Model | Accuracy | ROC-AUC |
| :--- | :--- | :--- |
| Logistic Regression | $\approx 70\%$ | $\approx 74\%$ |
| Random Forest | $\approx 74\%$ | $\approx 78\%$ |
| **XGBoost** | **$\approx 75\%$** | **$\approx 80\%$** |

### Key Insights
* **Glucose**, **BMI**, and **Age** are consistently the most influential features for diabetes prediction, as confirmed by both feature importance and SHAP analysis.
* The **XGBoost** model provides the best balance of performance and robustness.
* SHAP explanations make the black-box predictions clinically **interpretable**, highlighting *why* a specific patient was predicted to be high-risk.

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Healthcare-Diabetes-Risk-Prediction.git](https://github.com/YourUsername/Healthcare-Diabetes-Risk-Prediction.git)
    cd Healthcare-Diabetes-Risk-Prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Open and run the notebook:**
    ```bash
    notebooks/Diabetes_Risk_Prediction-HealthcareAI.ipynb
    ```
    *Run all cells* in the notebook to replicate the analysis, training, evaluation, and SHAP explanations.

---

## Academic & Practical Relevance
* Demonstrates an **end-to-end ML system design** in a critical domain.
* Applies **Explainable AI (XAI)** in healthcare, a mandatory requirement for real-world clinical deployment.
* Suitable for:
    * MS in Data Science / MS in Artificial Intelligence portfolios.
    * Healthcare AI / Biomedical Informatics programs.

### Future Enhancements
* Hyperparameter tuning (e.g., GridSearch or Bayesian Optimisation) for the XGBoost model.
* Model calibration and optimisation of the prediction threshold.
* Web application deployment (e.g., using Streamlit or Flask) to demonstrate a functional app.
* Extension to multi-disease risk prediction using similar clinical datasets.
