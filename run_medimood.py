#!/usr/bin/env python3
"""
MEDIMOOD Setup and Run Script
This script will train the models and start the web application
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def install_requirements():
    """Install required packages"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'flask', 'matplotlib', 'seaborn'
    ]
    
    print("📦 Installing required packages...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"🔄 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def create_folder_structure():
    """Create necessary folders"""
    folders = ['templates', 'static', 'models']
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📁 Created folder: {folder}")

def check_datasets():
    """Check if datasets exist"""
    heart_file = 'heart_dataset.csv'
    mental_file = 'mental_health_dataset.csv'
    
    if not os.path.exists(heart_file):
        print(f"❌ {heart_file} not found!")
        print("Please make sure you have uploaded the heart dataset file.")
        return False
        
    if not os.path.exists(mental_file):
        print(f"❌ {mental_file} not found!")
        print("Please make sure you have uploaded the mental health dataset file.")
        return False
        
    print("✅ Both dataset files found!")
    return True

def train_models():
    """Train both models"""
    print("\n🤖 Training Machine Learning Models...")
    
    # Load heart disease dataset
    print("📊 Loading heart disease data...")
    heart_data = pd.read_csv('heart_dataset.csv')
    print(f"   Shape: {heart_data.shape}")
    
    # Train heart model
    print("❤️ Training heart disease model...")
    X_heart = heart_data.drop('target', axis=1)
    y_heart = heart_data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
    
    heart_scaler = StandardScaler()
    X_train_scaled = heart_scaler.fit_transform(X_train)
    X_test_scaled = heart_scaler.transform(X_test)
    
    heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
    heart_model.fit(X_train_scaled, y_train)
    
    heart_accuracy = accuracy_score(y_test, heart_model.predict(X_test_scaled))
    print(f"   Heart model accuracy: {heart_accuracy:.2%}")
    
    # Save heart model
    pickle.dump(heart_model, open('heart_model.pkl', 'wb'))
    pickle.dump(heart_scaler, open('heart_scaler.pkl', 'wb'))
    
    # Load mental health dataset
    print("🧠 Loading mental health data...")
    mental_data = pd.read_csv('mental_health_dataset.csv')
    print(f"   Shape: {mental_data.shape}")
    
    # Train mental health model
    print("🧠 Training mental health model...")
    X_mental = mental_data.drop('mental_health_risk', axis=1)
    y_mental = mental_data['mental_health_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X_mental, y_mental, test_size=0.2, random_state=42)
    
    mental_scaler = StandardScaler()
    X_train_scaled = mental_scaler.fit_transform(X_train)
    X_test_scaled = mental_scaler.transform(X_test)
    
    mental_model = RandomForestClassifier(n_estimators=100, random_state=42)
    mental_model.fit(X_train_scaled, y_train)
    
    mental_accuracy = accuracy_score(y_test, mental_model.predict(X_test_scaled))
    print(f"   Mental health model accuracy: {mental_accuracy:.2%}")
    
    # Save mental health model
    pickle.dump(mental_model, open('mental_model.pkl', 'wb'))
    pickle.dump(mental_scaler, open('mental_scaler.pkl', 'wb'))
    
    print("\n✅ Models trained and saved successfully!")
    return True

def create_html_file():
    """Create the HTML template in templates folder"""
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # The HTML content is already created as an artifact
    # You'll need to copy the HTML artifact content to templates/index.html
    print("📝 Please copy the HTML template to templates/index.html")

def test_models():
    """Test the trained models"""
    print("\n🧪 Testing models...")
    
    try:
        # Test heart model
        heart_model = pickle.load(open('heart_model.pkl', 'rb'))
        heart_scaler = pickle.load(open('heart_scaler.pkl', 'rb'))
        
        # Sample prediction
        sample_heart = np.array([[50, 1, 2, 140, 250, 0, 1, 150, 0, 1.5]])
        sample_scaled = heart_scaler.transform(sample_heart)
        heart_pred = heart_model.predict(sample_scaled)[0]
        heart_prob = heart_model.predict_proba(sample_scaled)[0]
        
        print(f"❤️ Heart model test: Risk = {'HIGH' if heart_pred == 1 else 'LOW'}, Confidence = {max(heart_prob):.1%}")
        
        # Test mental health model
        mental_model = pickle.load(open('mental_model.pkl', 'rb'))
        mental_scaler = pickle.load(open('mental_scaler.pkl', 'rb'))
        
        # Sample prediction
        sample_mental = np.array([[30, 1, 7, 5, 6, 5, 6, 7, 5, 5]])
        sample_scaled = mental_scaler.transform(sample_mental)
        mental_pred = mental_model.predict(sample_scaled)[0]
        mental_prob = mental_model.predict_proba(sample_scaled)[0]
        
        risk_levels = ["LOW", "MEDIUM", "HIGH"]
        print(f"🧠 Mental health model test: Risk = {risk_levels[mental_pred]}, Confidence = {max(mental_prob):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {str(e)}")
        return False

def start_flask_app():
    """Start the Flask application"""
    print("\n🚀 Starting MEDIMOOD web application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⚡ Press Ctrl+C to stop the server")
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError:
        print("❌ app.py not found. Please make sure you have the Flask application file.")
    except Exception as e:
        print(f"❌ Error starting Flask app: {str(e)}")

def main():
    """Main setup function"""
    print("🏥 MEDIMOOD Setup Script")
    print("=" * 50)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Create folder structure
    create_folder_structure()
    
    # Step 3: Check datasets
    if not check_datasets():
        print("\n❌ Setup failed. Please upload the required CSV files.")
        return
    
    # Step 4: Train models
    if not train_models():
        print("\n❌ Model training failed.")
        return
    
    # Step 5: Test models
    if not test_models():
        print("\n❌ Model testing failed.")
        return
    
    # Step 6: Instructions
    print("\n🎉 MEDIMOOD Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Copy the HTML template to templates/index.html")
    print("2. Save the Flask app code as app.py")
    print("3. Run 'python app.py' to start the web server")
    print("4. Open http://localhost:5000 in your browser")
    
    # Ask if user wants to start the app
    start_app = input("\n❓ Do you want to start the web application now? (y/n): ")
    if start_app.lower() in ['y', 'yes']:
        start_flask_app()

if __name__ == "__main__":
    main()