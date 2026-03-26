# 📊 Customer Churn Prediction Web App

A Machine Learning based web application that predicts whether a telecom customer is likely to churn or continue using the service. This project combines **data analysis, model building, and web deployment using Flask** to deliver real-time predictions through an interactive UI.

---

## 🚀 Features

- 🔍 Predict customer churn in real-time  
- 🤖 Machine Learning model using **Random Forest**  
- ⚖️ Handles class imbalance using **SMOTE**  
- 🌐 Flask-based web application  
- 🎯 Displays prediction with confidence score  
- 💻 Clean and responsive user interface  

---

## 🧠 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Flask**
- **HTML, CSS, Bootstrap**

---

## 📂 Project Structure
Customer_Churn_Project/
│
├── app.py # Flask backend
├── model.sav # Trained ML model
├── columns.pkl # Feature columns
├── first_telc.csv # Dataset
│
├── templates/
│ └── home.html # Frontend UI
│
├── static/
│ └── style.css # Styling
│
└── notebook.ipynb # EDA + Model Training


---

## ⚙️ Installation & Setup

### 1. Clone the repository
``` bash
[git clone <your-repo-link>
cd Customer_Churn_Project
](https://github.com/Suryansh-Soni/Customer-Churn-Prediction-Web-App.git)
```

## Install dependencies
pip install flask pandas numpy scikit-learn

## Run the application
python app.py

## Open in browser
http://127.0.0.1:5000/

## Model Details
Algorithm: Random Forest Classifier
Technique: SMOTE (for handling class imbalance)
Input: Customer demographic and service details
Output:
Churn Prediction (Yes/No)
Probability Score

## Workflow
Data Collection (Telco dataset)
Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Model Training (Random Forest + SMOTE)
Model Evaluation
Deployment using Flask

## Use Case

This project helps telecom companies to:

Identify customers likely to churn
Take preventive actions
Improve customer retention
Enhance business decision-making

## Future Improvements
📈 Add interactive dashboard
🌍 Deploy on cloud (Render / AWS)
📱 Improve mobile responsiveness
🔄 Use ML pipeline for better scalability

 ##Author

Suryansh Soni

⭐ If you like this project , Give it a star ⭐ on GitHub!
