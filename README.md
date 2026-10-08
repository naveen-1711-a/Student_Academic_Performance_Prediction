# Student Academic Performance Prediction

An end-to-end Machine Learning project to predict academic outcomes and understand factors associated with student performance in the Education / EdTech industry. 

## 🏢 Business Scenario
An educational institution wants to predict academic outcomes and understand factors associated with student performance. The goal is to proactively identify at-risk students so that support can be targeted to learners who need it most, ultimately improving retention and academic success rates.

## 📊 Dataset
**Source**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

**Dataset Description**: Student demographic, social, and academic attributes with performance outcomes.
- Performance is categorized into:
  - **Low**: Grade < 10 (High Risk)
  - **Medium**: 10 ≤ Grade < 15 (Moderate Risk)
  - **High**: Grade ≥ 15 (Low Risk)

## 🧠 Data Quality & Preprocessing
To ensure an industry-standard pipeline, the data undergoes rigorous quality checks and preprocessing:
1. **Data Quality Check**: We actively scan for missing values, identify duplicate records, and use the IQR (Interquartile Range) method to detect and report outliers in numerical features.
2. **Preprocessing**:
   - Missing numerical values are imputed using the **median**, and categorical missing values are imputed using the **most frequent** strategy.
   - Numerical features are scaled using **StandardScaler**.
   - Categorical features are encoded using **One-Hot Encoding**.
3. **Dimensionality Reduction**: We apply **Principal Component Analysis (PCA)** to retain 95% of the variance. Since One-Hot Encoding drastically expands the feature space, PCA effectively reduces noise, mitigates the curse of dimensionality, and improves model training efficiency.

## 🤖 Models Built
We trained and evaluated several machine learning models to find the best fit for this problem:
- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**

### 🏆 Best Model Selection
Because predicting at-risk students correctly is critical (imbalanced cost of false negatives), models are evaluated using **Accuracy, Precision, Recall, and the F1 Score**. The pipeline automatically compares these metrics and saves the model with the highest **F1 Score** (`student_performance_model.pkl`) to ensure a balance between correctly identifying struggling students and minimizing false alarms.

## 💡 Business Insights & Actionable Recommendations
Based on the predictive model's findings, the educational institution can implement the following targeted actions:

1. **For "Low" Performance Predictions (High-Risk):**
   - **Action:** Immediately trigger an alert to academic advisors and the student's primary teachers.
   - **Intervention:** Assign a dedicated academic mentor, monitor attendance closely, and enroll the student in mandatory supplemental tutoring or after-school support programs.

2. **For "Medium" Performance Predictions (Moderate-Risk):**
   - **Action:** Add the student to a regular monitoring list.
   - **Intervention:** Send automated check-in emails with study resources and encourage voluntary participation in study groups. Counselors should schedule a mid-term check-in.

3. **For "High" Performance Predictions (Low-Risk):**
   - **Action:** Acknowledge their success and keep them engaged.
   - **Intervention:** Offer advanced learning opportunities, recommend them for peer tutoring roles (which helps struggling students), or invite them to specialized academic clubs.

## 🚀 Features & Applications
- **Automated ML Pipeline**: End-to-end preprocessing, training, and evaluation script (`src/main.py`).
- **Desktop GUI**: Built with `tkinter` for local administrative staff to make single-student predictions (`app.py`).
- **Web Application**: Built with `FastAPI`, providing a REST API and a responsive web frontend for integration into existing EdTech portals (`app.py`).

## 📁 Directory Structure
```text
student-performance-prediction/
├── data/
│   └── student-mat.csv               # Dataset used for training
├── models/
│   ├── student_performance_model.pkl # Best trained ML model
│   └── input_schema.json             # Input validation schema
├── outputs/                          # Generated plots and evaluation results
├── src/
│   ├── main.py                       # ML pipeline (training, PCA & evaluation)
│   └── predict.py                    # Prediction script
├── app.py                            # Tkinter GUI & FastAPI Web App
├── test_app.py                       # Unit tests
├── requirements.txt                  # Python dependencies
└── README.md
```

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash and git clone
  
   git clone https://github.com/naveen-1711-a/Student_Academic_Performance_Prediction.git
   cd Student_Academic_Performance_Prediction/student-performance-prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the Model:**
   ```bash
   python src/main.py
   ```
4. **Run the Desktop App:**
   ```bash
   python app.py
   ```
5. **Run the Web API:**
   ```bash
   uvicorn app:app --reload
   ```
