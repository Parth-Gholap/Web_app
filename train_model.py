"""
Complete Soybean Disease Classification System
Trains models to detect healthy vs Septoria brown spot disease
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2
import pickle
import warnings
warnings.filterwarnings('ignore')

class SoybeanDiseaseClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def extract_features(self, image_path):
        """
        Extract comprehensive features from leaf images
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize for consistent processing
        img_resized = cv2.resize(img, (224, 224))
        
        features = []
        
        # === COLOR FEATURES ===
        # RGB statistics
        for channel in range(3):
            channel_data = img_resized[:,:,channel].flatten()
            features.extend([
                np.mean(channel_data),    # Mean
                np.std(channel_data),     # Standard deviation
                np.median(channel_data),  # Median
                np.percentile(channel_data, 25),  # Q1
                np.percentile(channel_data, 75),  # Q3
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
        
        # === TEXTURE FEATURES ===
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern approximation
        lbp_like = self._calculate_lbp_features(gray)
        features.extend(lbp_like)
        
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
        
        # === SHAPE/MORPHOLOGY FEATURES ===
        # Contour-based features
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
        
        # === DISEASE-SPECIFIC FEATURES ===
        # Brown spot detection (specific for Septoria)
        brown_spots = self._detect_brown_spots(img_resized)
        features.extend(brown_spots)
        
        # Green health indicators
        green_health = self._calculate_green_health(img_resized)
        features.extend(green_health)
        
        return np.array(features)
    
    def _calculate_lbp_features(self, gray):
        """
        Calculate Local Binary Pattern-like features
        """
        features = []
        
        # Simple texture measures using different kernel sizes
        for kernel_size in [3, 5, 7]:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Morphological operations
            opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            features.extend([
                np.mean(opening),
                np.std(opening),
                np.mean(closing),
                np.std(closing),
            ])
        
        return features
    
    def _detect_brown_spots(self, img):
        """
        Detect brown spots characteristic of Septoria disease
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define brown color ranges
        brown_ranges = [
            ([5, 50, 20], [25, 255, 150]),    # Brown spots
            ([10, 40, 30], [20, 255, 120]),   # Dark brown
            ([0, 30, 20], [15, 255, 100]),    # Reddish brown
        ]
        
        total_brown_pixels = 0
        brown_regions = 0
        
        for lower, upper in brown_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            brown_pixels = np.sum(mask > 0)
            total_brown_pixels += brown_pixels
            
            # Count separate brown regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            brown_regions += len([c for c in contours if cv2.contourArea(c) > 10])
        
        # Calculate brown spot features
        total_pixels = img.shape[0] * img.shape[1]
        brown_percentage = total_brown_pixels / total_pixels
        
        return [brown_percentage, brown_regions, total_brown_pixels]
    
    def _calculate_green_health(self, img):
        """
        Calculate green health indicators
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Healthy green ranges
        healthy_green_ranges = [
            ([25, 40, 40], [85, 255, 255]),   # Normal green
            ([30, 50, 50], [80, 255, 200]),   # Vibrant green
        ]
        
        total_green_pixels = 0
        green_intensity_sum = 0
        
        for lower, upper in healthy_green_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            green_pixels = np.sum(mask > 0)
            total_green_pixels += green_pixels
            
            # Calculate average intensity in green regions
            if green_pixels > 0:
                green_regions = img[mask > 0]
                green_intensity_sum += np.sum(green_regions)
        
        total_pixels = img.shape[0] * img.shape[1]
        green_percentage = total_green_pixels / total_pixels
        green_intensity_avg = green_intensity_sum / max(total_green_pixels, 1)
        
        return [green_percentage, green_intensity_avg]
    
    def create_feature_names(self):
        """
        Create descriptive names for all features
        """
        names = []
        
        # RGB features
        rgb_channels = ['R', 'G', 'B']
        rgb_stats = ['mean', 'std', 'median', 'q25', 'q75']
        for channel in rgb_channels:
            for stat in rgb_stats:
                names.append(f'RGB_{channel}_{stat}')
        
        # HSV features
        hsv_channels = ['H', 'S', 'V']
        hsv_stats = ['mean', 'std', 'median']
        for channel in hsv_channels:
            for stat in hsv_stats:
                names.append(f'HSV_{channel}_{stat}')
        
        # LAB features
        lab_channels = ['L', 'A', 'B']
        lab_stats = ['mean', 'std']
        for channel in lab_channels:
            for stat in lab_stats:
                names.append(f'LAB_{channel}_{stat}')
        
        # Texture features
        for kernel_size in [3, 5, 7]:
            names.extend([
                f'texture_opening_mean_{kernel_size}',
                f'texture_opening_std_{kernel_size}',
                f'texture_closing_mean_{kernel_size}',
                f'texture_closing_std_{kernel_size}',
            ])
        
        # Edge and gradient features
        names.extend([
            'edge_density',
            'gradient_mean',
            'gradient_std',
            'gradient_max',
        ])
        
        # Shape features
        names.extend([
            'contour_area',
            'contour_perimeter',
            'contour_points',
        ])
        
        # Disease-specific features
        names.extend([
            'brown_spot_percentage',
            'brown_spot_regions',
            'brown_spot_pixels',
            'green_health_percentage',
            'green_intensity_avg',
        ])
        
        self.feature_names = names
        return names
    
    def load_dataset(self):
        """
        Load and prepare the dataset
        """
        print("Loading dataset...")
        
        # Check if main dataset path exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")
        
        print(f"Dataset directory found: {self.dataset_path}")
        print(f"Contents: {os.listdir(self.dataset_path)}")
        
        categories = ["healthy", "Septoria brown spot"]
        X = []
        y = []
        image_paths = []
        
        for category in categories:
            category_path = os.path.join(self.dataset_path, category)
            
            if not os.path.exists(category_path):
                print(f"Warning: Directory {category_path} not found!")
                continue
            
            print(f"Processing {category}...")
            
            # Get all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_files.extend([f for f in os.listdir(category_path) if f.lower().endswith(ext)])
            
            print(f"  Found {len(image_files)} images")
            
            if len(image_files) == 0:
                print(f"  Warning: No image files found in {category_path}")
                continue
            
            for i, filename in enumerate(image_files):
                if i % 50 == 0:
                    print(f"    Processing {i+1}/{len(image_files)}")
                
                image_path = os.path.join(category_path, filename)
                features = self.extract_features(image_path)
                
                if features is not None:
                    X.append(features)
                    y.append(category)
                    image_paths.append(image_path)
                else:
                    print(f"    Failed to process: {filename}")
        
        # Convert to numpy arrays
        if len(X) == 0:
            raise ValueError("No images were successfully processed! Please check your dataset directory structure and image files.")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
        print(f"Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for class_name, count in zip(unique, counts):
            print(f"  {class_name}: {count} samples")
        
        # Check for minimum samples per class
        if len(unique) < 2:
            raise ValueError("Need at least 2 classes for classification!")
        
        if np.min(counts) < 5:
            print("Warning: Some classes have very few samples. Consider collecting more data.")
        
        return X, y, image_paths
    
    def train_models(self, X, y):
        """
        Train multiple models and compare performance
        """
        print("\nTraining models...")
        
        # Create feature names
        self.create_feature_names()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                if name in ['SVM', 'Logistic Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                if name in ['SVM', 'Logistic Regression']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'y_test': y_test,
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        if len(results) == 0:
            raise RuntimeError("No models were successfully trained!")
        
        self.models = results
        self.X_test_scaled = X_test_scaled
        self.X_test = X_test
        
        return results
    
    def evaluate_models(self):
        """
        Evaluate and compare all trained models
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        best_model = None
        best_accuracy = 0
        
        for name, result in self.models.items():
            comparison_data.append({
                'Model': name,
                'Test Accuracy': result['accuracy'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std'],
            })
            
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = name
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        print(comparison_df.to_string(index=False))
        
        print(f"\nBest Model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        # Detailed evaluation of best model
        print(f"\nDetailed evaluation of {best_model}:")
        print("-" * 40)
        
        best_result = self.models[best_model]
        
        # Classification report
        class_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(best_result['y_test'], best_result['y_pred'], 
                                  target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        
        plt.figure(figsize=(12, 5))
        
        # Confusion matrix heatmap
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {best_model}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Model comparison bar plot
        plt.subplot(1, 2, 2)
        models = [item['Model'] for item in comparison_data]
        accuracies = [item['Test Accuracy'] for item in comparison_data]
        
        bars = plt.bar(models, accuracies, color=['green' if acc == max(accuracies) else 'skyblue' for acc in accuracies])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Test Accuracy')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return best_model, best_result
    
    def analyze_feature_importance(self, best_model_name):
        """
        Analyze which features are most important for classification
        """
        print(f"\nFeature Importance Analysis for {best_model_name}")
        print("-" * 50)
        
        model = self.models[best_model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 15 Most Important Features:")
            print(feature_importance.head(15).to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importances - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        elif hasattr(model, 'coef_'):
            # Linear models
            coefficients = np.abs(model.coef_[0])
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': coefficients
            }).sort_values('importance', ascending=False)
            
            print("Top 15 Most Important Features (by coefficient magnitude):")
            print(feature_importance.head(15).to_string(index=False))
    
    def save_best_model(self, best_model_name, save_path="best_soybean_model.pkl"):
        """
        Save the best model for future use
        """
        print(f"\nSaving best model ({best_model_name}) to {save_path}")
        
        model_package = {
            'model': self.models[best_model_name]['model'],
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_name': best_model_name,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Model saved successfully!")
        return save_path
    
    def predict_single_image(self, image_path, model_path="best_soybean_model.pkl"):
        """
        Predict disease for a single image
        """
        # Load model
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        model = model_package['model']
        scaler = model_package['scaler']
        label_encoder = model_package['label_encoder']
        model_name = model_package['model_name']
        
        # Extract features
        features = self.extract_features(image_path)
        if features is None:
            return None
        
        # Prepare features
        features = features.reshape(1, -1)
        
        if model_name in ['SVM', 'Logistic Regression']:
            features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Decode prediction
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        return {
            'predicted_class': predicted_class,
            'confidence': probability[prediction],
            'probabilities': dict(zip(label_encoder.classes_, probability))
        }

def run_complete_pipeline():
    """
    Run the complete training pipeline
    """
    # Configuration - FIXED DATASET PATH
    dataset_path = r"C:\Users\Parth\OneDrive\Desktop\Soybean_Dataset_Processed"  # Corrected path
    
    print("SOYBEAN DISEASE CLASSIFICATION PIPELINE")
    print("="*60)
    print(f"Dataset path: {dataset_path}")
    print("="*60)
    
    try:
        # Initialize classifier
        classifier = SoybeanDiseaseClassifier(dataset_path)
        
        # Load dataset
        X, y, image_paths = classifier.load_dataset()
        
        # Train models
        results = classifier.train_models(X, y)
        
        # Evaluate models
        best_model_name, best_result = classifier.evaluate_models()
        
        # Analyze feature importance
        classifier.analyze_feature_importance(best_model_name)
        
        # Save best model
        model_path = classifier.save_best_model(best_model_name)
        
        print(f"\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Best model: {best_model_name}")
        print(f"Test accuracy: {best_result['accuracy']:.4f}")
        print(f"Model saved to: {model_path}")
        print("="*60)
        
        return classifier, model_path
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check that the dataset directory exists")
        print("2. Verify that 'healthy' and 'Septoria brown spot' folders exist")
        print("3. Make sure there are image files in both folders")
        print("4. Check that image files have valid extensions (.jpg, .png, etc.)")
        raise

if __name__ == "__main__":
    classifier, model_path = run_complete_pipeline()