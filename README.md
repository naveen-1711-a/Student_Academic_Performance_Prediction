# Student Academic Performance Prediction

An end-to-end Machine Learning project to predict student academic performance based on demographic, social, and school-related features.

## 🚀 Features
- **Machine Learning Pipeline**: Trains and evaluates multiple models (Logistic Regression, Random Forest, Gradient Boosting, SVM).
- **Automated Model Selection**: Automatically selects and saves the best-performing model based on F1 Score.
- **Two User Interfaces**:
  - **Desktop GUI**: Built with `tkinter` for local predictions.
  - **Web Application**: Built with `FastAPI`, providing a REST API and a responsive web frontend.
- **Actionable Recommendations**: Provides tailored advice based on the predicted performance level (Low, Medium, High).

## 📁 Directory Structure
```text
student-performance-prediction/
├── data/
│   └── student-mat.csv               # Dataset used for training
├── models/
│   ├── student_performance_model.pkl # Best trained ML model
│   └── input_schema.json             # Input validation schema
├── outputs/                          # Generated plots and evaluation results
│   ├── confusion_matrix.png
│   ├── model_f1_comparison.png
│   ├── performance_distribution.png
│   └── model_results.csv / .json
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

## 🧠 Training the Model

If you wish to retrain the model or test on a different dataset:
```bash
python src/main.py
```
This will:
- Load and preprocess the dataset.
- Train several classifier models.
- Compare them and save the best one as a `.pkl` file in `models/`.
- Save evaluation plots and metrics in the `outputs/` folder.

## 💻 Usage

### 1. Run the Desktop GUI (Tkinter)
To launch the standalone desktop application:
```bash
python app.py
```
*A window will appear allowing you to input student details and get a prediction.*

### 2. Run the Web App & API (FastAPI)
To serve the web application and REST API locally:
```bash
uvicorn app:app --reload
```
- **Web UI**: Open your browser and navigate to [http://localhost:8000/](http://localhost:8000/)
- **API Docs (Swagger)**: View the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)

## 📊 Dataset
The dataset includes various features such as:
- Student grades, demographic, social and school related features.
- Performance is categorized into:
  - **Low**: Grade < 10
  - **Medium**: 10 ≤ Grade < 15
  - **High**: Grade ≥ 15