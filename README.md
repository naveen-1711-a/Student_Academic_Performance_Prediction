# Student Academic Performance Prediction

An end-to-end Machine Learning project to predict student academic performance based on demographic, social, and school-related features.

## 📊 Dataset
**Source**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

The dataset includes various features such as:
- Student grades, demographic, social and school related features.
- Performance is categorized into:
  - **Low**: Grade < 10
  - **Medium**: 10 ≤ Grade < 15
  - **High**: Grade ≥ 15

## 🧠 Models Used & Selection
The pipeline trains and evaluates the following Machine Learning models:
- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**

### 🏆 Best Model
The training script automatically compares these models and selects the one with the highest **F1 Score** to handle class imbalances. Typically, ensemble methods like **Random Forest** or **Gradient Boosting** perform the best on this dataset and are selected as the final model (`student_performance_model.pkl`) for predictions.

### 🔍 PCA Feature Selection (Concepts & Application)
Principal Component Analysis (PCA) can be applied to this dataset to reduce dimensionality and perform feature selection:
- **How it works:** PCA transforms the original features into a new set of uncorrelated variables (principal components) that capture the maximum variance in the data.
- **Why use it:** After One-Hot Encoding categorical variables, the feature space grows significantly. PCA helps in reducing noise, preventing overfitting, and speeding up model training while retaining the most important information (e.g., keeping 95% of the variance).

## 🚀 Features
- **Machine Learning Pipeline**: Automated preprocessing, training, and evaluation.
- **Two User Interfaces**:
  - **Desktop GUI**: Built with `tkinter` for local predictions.
  - **Web Application**: Built with `FastAPI`, providing a REST API and a responsive web frontend.
- **Actionable Recommendations**: Provides tailored advice based on the predicted performance level.

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
│   ├── main.py                       # ML pipeline (training & evaluation)
│   └── predict.py                    # Prediction script
├── app.py                            # Application (Tkinter GUI & FastAPI Web App)
├── test_app.py                       # Unit tests
├── requirements.txt                  # Python dependencies
└── README.md
```

## 🛠️ Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/naveen-1711-a/Student_Academic_Performance_Prediction.git
   cd Student_Academic_Performance_Prediction/student-performance-prediction
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Training the Model
```bash
python src/main.py
```
This script loads the dataset, trains all the models, compares them, and saves the best one.

### 2. Run the Desktop GUI (Tkinter)
```bash
python app.py
```

### 3. Run the Web App & API (FastAPI)
```bash
uvicorn app:app --reload
```
- **Web UI**: Open your browser at [http://localhost:8000/](http://localhost:8000/)
- **API Docs (Swagger)**: View API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)