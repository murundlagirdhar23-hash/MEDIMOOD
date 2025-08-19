# 🏥 MEDIMOOD - Health Risk Prediction System

A comprehensive machine learning project that predicts heart disease and mental health risks with personalized recommendations.

## 📋 Project Overview

MEDIMOOD is a user-friendly web application that:
- Predicts heart disease risk using clinical parameters
- Assesses mental health risk based on lifestyle factors
- Provides personalized health tips and recommendations
- Uses easy-to-understand language for common people
- Generates accurate predictions with confidence scores

## 🎯 Features

✅ **Heart Disease Prediction** - Analyzes 10 medical parameters
✅ **Mental Health Assessment** - Evaluates psychological well-being
✅ **User-Friendly Interface** - Simple terms, no medical jargon
✅ **Personalized Tips** - Custom recommendations based on results
✅ **High Accuracy Models** - Trained with Random Forest algorithm
✅ **Responsive Design** - Works on all devices
✅ **Real-time Predictions** - Instant results with confidence scores

## 📁 Project Structure

```
MEDIMOOD/
│
├── heart_dataset.csv              # Heart disease data (your file)
├── mental_health_dataset.csv      # Mental health data (your file)
├── app.py                        # Flask web application
├── run_medimood.py              # Setup and training script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── templates/
│   └── index.html              # Web interface
│
├──                   # Trained models (generated)
├── heart_model.pkl
├── heart_scaler.pkl
├── mental_model.pkl
├── mental_scaler.pkl
│
└── notebooks/                  # Jupyter notebooks (optional)
    └── medimood_analysis.ipynb
```

## 🚀 Quick Start Guide

### Step 1: Prepare Your Environment

1. **Install Python** (version 3.7 or higher)
2. **Create a project folder**:
   ```bash
   mkdir medimood_project
   cd medimood_project
   ```

### Step 2: Set Up Files

1. **Save the provided code files**:
   - Copy the Jupyter notebook code → save as `medimood_training.ipynb`
   - Copy the Flask app code → save as `app.py`
   - Copy the HTML template → save as `templates/index.html`
   - Copy the setup script → save as `run_medimood.py`

2. **Upload your CSV files**:
   - Place `heart_dataset.csv` in the project folder
   - Place `mental_health_dataset.csv` in the project folder

### Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn flask matplotlib seaborn
```

Or create `requirements.txt`:
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
flask>=2.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 4: Train the Models

**Option A: Using Jupyter Notebook**
1. Open `medimood_training.ipynb` in Jupyter
2. Run all cells to train and save models
3. This will create `.pkl` files for the trained models

**Option B: Using Python Script**
```bash
python run_medimood.py
```

### Step 5: Create Templates Folder

```bash
mkdir templates
```
Then save the HTML code as `templates/index.html`

### Step 6: Run the Web Application

```bash
python app.py
```

The application will start at: **http://localhost:5000**

## 📊 Dataset Information

### Heart Disease Dataset
- **Columns**: 11 features + 1 target
- **Features**: Age, gender, chest pain type, blood pressure, cholesterol, etc.
- **Target**: 0 = No heart disease, 1 = Heart