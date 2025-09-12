"""
Flask Web Application for Soybean Disease Prediction
Upload images and get disease predictions with cure information
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class SoybeanDiseasePredictor:
    def __init__(self, model_path="best_soybean_model.pkl"):
        self.model_path = model_path
        self.load_model()
        self.cure_info = self.get_septoria_cure_info()
    
    def get_septoria_cure_info(self):
        return {
            'title': 'SEPTORIA BROWN SPOT MANAGEMENT',
            'treatments': [
                'Apply a Recommended Fungicide',
                'Implement Crop Rotation',
                'Use Tillage Practices',
                'Select Resistant Soybean Varieties',
                'Improve Air Circulation'
            ],
            'reference': 'For detailed reference, visit Crop Protection Network: https://cropprotectionnetwork.org/publications/septoria-brown-spot-of-soybean',
            'immediate_actions': [
                'Remove and destroy affected leaves',
                'Apply appropriate fungicide immediately',
                'Monitor surrounding plants for spread',
                'Ensure proper plant spacing for air circulation'
            ],
            'prevention_tips': [
                'Plant resistant varieties when available',
                'Rotate crops (avoid soybeans for 2-3 years)',
                'Use clean tillage equipment',
                'Avoid overhead irrigation',
                'Remove crop residue'
            ],
            'fungicide_guidelines': [
                'Apply at first sign of symptoms',
                'Follow label instructions carefully',
                'Reapply as recommended',
                'Time applications during cool, wet periods'
            ]
        }
    
    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.label_encoder = model_package['label_encoder']
            self.feature_names = model_package['feature_names']
            self.model_name = model_package['model_name']
            
        except FileNotFoundError:
            raise Exception(f"Model file not found: {self.model_path}")
    
    def extract_features(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_resized = cv2.resize(img, (224, 224))
        features = []
        
        # RGB statistics
        for channel in range(3):
            channel_data = img_resized[:,:,channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
            ])
        
        # HSV color features
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        for channel in range(3):
            channel_data = hsv[:,:,channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
            ])
        
        # LAB color features
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        for channel in range(3):
            channel_data = lab[:,:,channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
            ])
        
        # Texture features
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        for kernel_size in [3, 5, 7]:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            features.extend([
                np.mean(opening),
                np.std(opening),
                np.mean(closing),
                np.std(closing),
            ])
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude),
        ])
        
        # Shape features
        contours, _ = cv2.findContours(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features.extend([
                cv2.contourArea(largest_contour),
                cv2.arcLength(largest_contour, True),
                len(largest_contour),
            ])
        else:
            features.extend([0, 0, 0])
        
        # Disease-specific features
        brown_spots = self._detect_brown_spots(img_resized)
        features.extend(brown_spots)
        
        green_health = self._calculate_green_health(img_resized)
        features.extend(green_health)
        
        return np.array(features)
    
    def _detect_brown_spots(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        brown_ranges = [
            ([5, 50, 20], [25, 255, 150]),
            ([10, 40, 30], [20, 255, 120]),
            ([0, 30, 20], [15, 255, 100]),
        ]
        
        total_brown_pixels = 0
        brown_regions = 0
        
        for lower, upper in brown_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            brown_pixels = np.sum(mask > 0)
            total_brown_pixels += brown_pixels
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            brown_regions += len([c for c in contours if cv2.contourArea(c) > 10])
        
        total_pixels = img.shape[0] * img.shape[1]
        brown_percentage = total_brown_pixels / total_pixels
        
        return [brown_percentage, brown_regions, total_brown_pixels]
    
    def _calculate_green_health(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        healthy_green_ranges = [
            ([25, 40, 40], [85, 255, 255]),
            ([30, 50, 50], [80, 255, 200]),
        ]
        
        total_green_pixels = 0
        green_intensity_sum = 0
        
        for lower, upper in healthy_green_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            green_pixels = np.sum(mask > 0)
            total_green_pixels += green_pixels
            
            if green_pixels > 0:
                green_regions = img[mask > 0]
                green_intensity_sum += np.sum(green_regions)
        
        total_pixels = img.shape[0] * img.shape[1]
        green_percentage = total_green_pixels / total_pixels
        green_intensity_avg = green_intensity_sum / max(total_green_pixels, 1)
        
        return [green_percentage, green_intensity_avg]
    
    def predict_image(self, image_path):
        if not os.path.exists(image_path):
            return None, f"Image not found: {image_path}"
        
        features = self.extract_features(image_path)
        if features is None:
            return None, "Could not process image"
        
        features = features.reshape(1, -1)
        
        if self.model_name in ['SVM', 'Logistic Regression']:
            features = self.scaler.transform(features)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = float(probabilities[i])
        
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'is_healthy': predicted_class == 'healthy',
            'has_disease': predicted_class == 'Septoria brown spot',
            'cure_info': self.cure_info if predicted_class == 'Septoria brown spot' else None
        }
        
        return result, None

# Initialize the predictor
try:
    predictor = SoybeanDiseasePredictor()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR = str(e)

@app.route('/')
def index():
    return render_template('index.html', model_loaded=MODEL_LOADED, 
                         model_error=MODEL_ERROR if not MODEL_LOADED else None)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Make prediction
        result, error = predictor.predict_image(filepath)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Add image data to result
        result['image_data'] = f"data:image/jpeg;base64,{img_base64}"
        result['filename'] = filename
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)