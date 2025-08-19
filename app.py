from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)

# ========================================
# HELPER FUNCTIONS
# ========================================

def load_models():
    """Load all trained models with error handling"""
    try:
        print("Loading models...")
        
        # Check if model files exist
        model_files = ['heart_model.pkl', 'heart_scaler.pkl', 'mental_model.pkl', 'mental_scaler.pkl']
        missing_files = []
        
        for file in model_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"Missing model files: {missing_files}")
            return None, None, None, None
        
        # Load models
        heart_model = pickle.load(open('heart_model.pkl', 'rb'))
        heart_scaler = pickle.load(open('heart_scaler.pkl', 'rb'))
        mental_model = pickle.load(open('mental_model.pkl', 'rb'))
        mental_scaler = pickle.load(open('mental_scaler.pkl', 'rb'))
        
        print("All models loaded successfully!")
        return heart_model, heart_scaler, mental_model, mental_scaler
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print(traceback.format_exc())
        return None, None, None, None

def get_heart_tips(risk_level, confidence):
    """Get heart health tips"""
    if risk_level == 1:  # High risk
        tips = [
            "🚨 Please consult a doctor immediately for proper check-up",
            "💊 Follow your doctor's medication advice strictly",
            "🚭 Quit smoking and avoid secondhand smoke",
            "🥗 Eat heart-healthy foods: fruits, vegetables, whole grains",
            "🧂 Reduce salt intake to less than 1 teaspoon per day",
            "🏃‍♂️ Start with light exercise like walking (doctor approved)",
            "😌 Practice stress management: deep breathing, meditation",
            "⚖️ Maintain a healthy weight",
            "💧 Drink plenty of water daily",
            "📊 Monitor your blood pressure regularly"
        ]
        risk_text = "HIGH RISK"
        risk_color = "danger"
    else:  # Low risk
        tips = [
            "✅ Great news! Your heart health looks good",
            "🥗 Continue eating fruits and vegetables daily",
            "🏃‍♂️ Keep exercising for 30 minutes, 5 days a week",
            "🚭 Stay smoke-free and limit alcohol",
            "🧂 Keep using less salt in your food",
            "😴 Get 7-8 hours of good sleep each night",
            "💧 Drink 8 glasses of water daily",
            "📊 Check your blood pressure once a month",
            "😊 Maintain a positive lifestyle",
            "👨‍⚕️ Get regular health check-ups"
        ]
        risk_text = "LOW RISK"
        risk_color = "success"
    
    return tips, risk_text, risk_color

def get_mental_health_tips(risk_level, confidence):
    """Get mental health tips"""
    risk_levels = ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
    risk_colors = ["success", "warning", "danger"]
    
    if risk_level == 0:  # Low risk
        tips = [
            "✅ Excellent! Your mental health is in great shape",
            "🧘‍♀️ Keep practicing mindfulness and meditation",
            "👥 Continue nurturing your social relationships",
            "🏃‍♀️ Maintain regular physical activity",
            "😴 Keep your good sleep routine",
            "🎯 Set small, achievable daily goals",
            "🎨 Continue hobbies that make you happy",
            "📚 Keep learning new things",
            "🌞 Spend time outdoors in nature",
            "💪 You're doing great - keep it up!"
        ]
    elif risk_level == 1:  # Medium risk
        tips = [
            "⚠️ You might be experiencing some stress - that's normal",
            "🧘‍♀️ Try 10 minutes of deep breathing daily",
            "👥 Talk to friends or family about your feelings",
            "🚶‍♀️ Take a 20-minute walk outside every day",
            "📱 Take breaks from social media and news",
            "🎨 Try relaxing activities like drawing or music",
            "😴 Aim for 7-8 hours of sleep each night",
            "📝 Write down 3 things you're grateful for daily",
            "☕ Limit caffeine and alcohol",
            "💬 Consider talking to a counselor - it helps!"
        ]
    else:  # High risk
        tips = [
            "🚨 Please consider reaching out for professional support",
            "☎️ Contact a mental health helpline: 1-800-273-8255",
            "👨‍⚕️ Schedule an appointment with a mental health professional",
            "👥 Reach out to trusted friends or family members",
            "🧘‍♀️ Practice relaxation: deep breathing, meditation",
            "📝 Keep a daily mood journal to track patterns",
            "🏃‍♀️ Try gentle exercise like walking or yoga",
            "😴 Focus on getting regular, quality sleep",
            "🍎 Eat nutritious meals regularly",
            "💪 Remember: Asking for help shows strength, not weakness"
        ]
    
    return tips, risk_levels[risk_level], risk_colors[risk_level]

# ========================================
# ROUTES
# ========================================

@app.route('/')
def home():
    """Main page"""
    print("Home page accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions"""
    print("Predict endpoint called")
    
    try:
        # Load models
        heart_model, heart_scaler, mental_model, mental_scaler = load_models()
        
        if heart_model is None:
            print("Models not loaded - returning error")
            return jsonify({
                'error': 'Models not found. Please train the models first by running the Jupyter notebook.',
                'success': False,
                'details': 'Make sure heart_model.pkl, heart_scaler.pkl, mental_model.pkl, and mental_scaler.pkl exist in the project folder.'
            }), 400
        
        # Get form data
        data = request.json
        print(f"Received data: {data}")
        
        if not data:
            return jsonify({
                'error': 'No data received',
                'success': False
            }), 400
        
        # Validate required fields
        required_heart_fields = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol', 
                                'blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_pain', 'oldpeak']
        required_mental_fields = ['age', 'gender', 'stress_level', 'sleep_quality', 'social_support',
                                 'work_life_balance', 'anxiety_level', 'mood_changes', 'energy_level', 'concentration']
        
        missing_fields = []
        for field in required_heart_fields + required_mental_fields:
            if field not in data or data[field] == '':
                missing_fields.append(field)
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'success': False
            }), 400
        
        # Heart prediction
        print("Making heart prediction...")
        heart_input = np.array([[
            float(data['age']),
            int(data['sex']),
            int(data['chest_pain']),
            float(data['resting_bp']),
            float(data['cholesterol']),
            int(data['blood_sugar']),
            int(data['resting_ecg']),
            float(data['max_heart_rate']),
            int(data['exercise_pain']),
            float(data['oldpeak'])
        ]])
        
        heart_input_scaled = heart_scaler.transform(heart_input)
        heart_prediction = heart_model.predict(heart_input_scaled)[0]
        heart_probability = heart_model.predict_proba(heart_input_scaled)[0]
        heart_confidence = max(heart_probability)
        
        print(f"Heart prediction: {heart_prediction}, confidence: {heart_confidence}")
        
        # Mental health prediction
        print("Making mental health prediction...")
        mental_input = np.array([[
            float(data['age']),
            int(data['gender']),
            float(data['stress_level']),
            float(data['sleep_quality']),
            float(data['social_support']),
            float(data['work_life_balance']),
            float(data['anxiety_level']),
            float(data['mood_changes']),
            float(data['energy_level']),
            float(data['concentration'])
        ]])
        
        mental_input_scaled = mental_scaler.transform(mental_input)
        mental_prediction = mental_model.predict(mental_input_scaled)[0]
        mental_probability = mental_model.predict_proba(mental_input_scaled)[0]
        mental_confidence = max(mental_probability)
        
        print(f"Mental prediction: {mental_prediction}, confidence: {mental_confidence}")
        
        # Get tips
        heart_tips, heart_risk_text, heart_color = get_heart_tips(heart_prediction, heart_confidence)
        mental_tips, mental_risk_text, mental_color = get_mental_health_tips(mental_prediction, mental_confidence)
        
        result = {
            'success': True,
            'heart_risk': heart_risk_text,
            'heart_confidence': f"{heart_confidence:.1%}",
            'heart_color': heart_color,
            'heart_tips': heart_tips,
            'mental_risk': mental_risk_text,
            'mental_confidence': f"{mental_confidence:.1%}",
            'mental_color': mental_color,
            'mental_tips': mental_tips
        }
        
        print("Prediction successful, returning results")
        return jsonify(result)
        
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        return jsonify({
            'error': f'Invalid input data: {str(e)}',
            'success': False
        }), 400
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    print("Health check called")
    heart_model, heart_scaler, mental_model, mental_scaler = load_models()
    models_loaded = heart_model is not None
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'message': 'Medimood is running!' if models_loaded else 'Please train models first',
        'current_directory': os.getcwd(),
        'files_in_directory': os.listdir('.')
    })

@app.route('/test')
def test_page():
    """Test page to check if Flask is working"""
    return jsonify({
        'message': 'Flask server is running!',
        'status': 'OK'
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🏥 Starting MEDIMOOD Application...")
    print("📁 Current directory:", os.getcwd())
    print("📂 Files in directory:", os.listdir('.'))
    
    # Check if templates folder exists
    if not os.path.exists('templates'):
        print("❌ 'templates' folder not found!")
        print("Please create 'templates' folder and put index.html inside it")
    elif not os.path.exists('templates/index.html'):
        print("❌ 'templates/index.html' not found!")
        print("Please put the HTML template in templates/index.html")
    else:
        print("✅ Templates folder and index.html found")
    
    # Check for model files
    model_files = ['heart_model.pkl', 'heart_scaler.pkl', 'mental_model.pkl', 'mental_scaler.pkl']
    missing_models = [f for f in model_files if not os.path.exists(f)]
    
    if missing_models:
        print(f"⚠️ Missing model files: {missing_models}")
        print("Please run the Jupyter notebook first to train and save the models")
    else:
        print("✅ All model files found")
    
    print("\n🚀 Server starting at: http://localhost:5000")
    print("⚡ Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)