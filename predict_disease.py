import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

class SoybeanDiseasePredictor:
    def __init__(self, model_path="best_soybean_model.pkl"):
        """
        Initialize predictor with trained model
        """
        self.model_path = model_path
        self.load_model()
        self.cure_info = self.get_septoria_cure_info()
    
    def get_septoria_cure_info(self):
        """
        Return cure information for Septoria brown spot
        """
        return {
            'title': 'SEPTORIA BROWN SPOT MANAGEMENT',
            'treatments': [
                'Apply a Recommended Fungicide',
                'Implement Crop Rotation',
                'Use Tillage Practices',
                'Select Resistant Soybean Varieties',
                'Improve Air Circulation'
            ],
            'reference': 'For detailed reference, visit Crop Protection Network:\nhttps://cropprotectionnetwork.org/publications/septoria-brown-spot-of-soybean',
            'immediate_actions': [
                '• Remove and destroy affected leaves',
                '• Apply appropriate fungicide immediately',
                '• Monitor surrounding plants for spread',
                '• Ensure proper plant spacing for air circulation'
            ]
        }
    
    def load_model(self):
        """
        Load the trained model and preprocessing objects
        """
        try:
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.label_encoder = model_package['label_encoder']
            self.feature_names = model_package['feature_names']
            self.model_name = model_package['model_name']
            
            print(f"Model loaded successfully: {self.model_name}")
            print(f"Classes: {self.label_encoder.classes_}")
            
        except FileNotFoundError:
            print(f"Model file not found: {self.model_path}")
            print("Please train the model first using the training script.")
            raise
    
    def extract_features(self, image_path):
        """
        Extract the same features used during training
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
        
        # === TEXTURE FEATURES ===
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Texture features using morphological operations
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
        
        # === SHAPE FEATURES ===
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
        # Brown spot detection
        brown_spots = self._detect_brown_spots(img_resized)
        features.extend(brown_spots)
        
        # Green health indicators
        green_health = self._calculate_green_health(img_resized)
        features.extend(green_health)
        
        return np.array(features)
    
    def _detect_brown_spots(self, img):
        """
        Detect brown spots characteristic of Septoria disease
        """
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
        """
        Calculate green health indicators
        """
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
        """
        Predict disease for a single image
        """
        if not os.path.exists(image_path):
            return None, f"Image not found: {image_path}"
        
        # Extract features
        features = self.extract_features(image_path)
        if features is None:
            return None, "Could not process image"
        
        # Prepare features
        features = features.reshape(1, -1)
        
        # Scale features if needed
        if self.model_name in ['SVM', 'Logistic Regression']:
            features = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Decode prediction
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = probabilities[i]
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prob_dict,
            'is_healthy': predicted_class == 'healthy',
            'has_disease': predicted_class == 'Septoria brown spot'
        }
        
        return result, None
    
    def predict_with_visualization(self, image_path, save_result=False, output_dir="predictions"):
        """
        Predict and visualize the result with cure information
        """
        result, error = self.predict_image(image_path)
        
        if error:
            print(f"Error: {error}")
            return None
        
        # Load and display image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualization with different layout for disease detection
        if result['has_disease']:
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle('SOYBEAN DISEASE DETECTION & TREATMENT', fontsize=16, fontweight='bold')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show original image
        if result['has_disease']:
            axes[0, 0].imshow(img_rgb)
            axes[0, 0].set_title(f'Input Image\n{os.path.basename(image_path)}', fontsize=12)
            axes[0, 0].axis('off')
            
            # Prediction results
            axes[0, 1].axis('off')
        else:
            axes[0].imshow(img_rgb)
            axes[0].set_title(f'Input Image\n{os.path.basename(image_path)}', fontsize=12)
            axes[0].axis('off')
            
            # Prediction results
            axes[1].axis('off')
        
        # Prediction text
        prediction_text = f"PREDICTION RESULTS\n\n"
        prediction_text += f"Predicted Class: {result['predicted_class']}\n"
        prediction_text += f"Confidence: {result['confidence']:.1%}\n\n"
        
        # Health status
        if result['is_healthy']:
            prediction_text += "✅ HEALTHY LEAF\n"
            prediction_text += "No disease detected\n\n"
        else:
            prediction_text += "⚠️  DISEASE DETECTED\n"
            prediction_text += "Septoria Brown Spot identified\n\n"
        
        # Probability breakdown
        prediction_text += "Class Probabilities:\n"
        for class_name, prob in result['probabilities'].items():
            prediction_text += f"• {class_name}: {prob:.1%}\n"
        
        # Basic recommendation
        if result['is_healthy']:
            prediction_text += "\nBasic Recommendations:\n"
            prediction_text += "• Continue regular monitoring\n"
            prediction_text += "• Maintain good plant health practices\n"
            prediction_text += "• Ensure proper nutrition and watering\n"
        
        # Display prediction text
        if result['has_disease']:
            axes[0, 1].text(0.05, 0.95, prediction_text, transform=axes[0, 1].transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
        else:
            axes[1].text(0.05, 0.95, prediction_text, transform=axes[1].transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        # Add cure information if disease is detected
        if result['has_disease']:
            # Treatment methods
            axes[1, 0].axis('off')
            cure_text = f"{self.cure_info['title']}\n\n"
            cure_text += "TREATMENT METHODS:\n"
            for i, treatment in enumerate(self.cure_info['treatments'], 1):
                cure_text += f"{i}. {treatment}\n"
            
            cure_text += "\nIMMEDIATE ACTIONS:\n"
            for action in self.cure_info['immediate_actions']:
                cure_text += f"{action}\n"
            
            axes[1, 0].text(0.05, 0.95, cure_text, transform=axes[1, 0].transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
            
            # Reference information
            axes[1, 1].axis('off')
            reference_text = "ADDITIONAL INFORMATION\n\n"
            reference_text += "Prevention Tips:\n"
            reference_text += "• Plant resistant varieties when available\n"
            reference_text += "• Rotate crops (avoid soybeans for 2-3 years)\n"
            reference_text += "• Use clean tillage equipment\n"
            reference_text += "• Avoid overhead irrigation\n"
            reference_text += "• Remove crop residue\n\n"
            
            reference_text += "Fungicide Application:\n"
            reference_text += "• Apply at first sign of symptoms\n"
            reference_text += "• Follow label instructions carefully\n"
            reference_text += "• Reapply as recommended\n"
            reference_text += "• Time applications during cool, wet periods\n\n"
            
            reference_text += f"{self.cure_info['reference']}"
            
            axes[1, 1].text(0.05, 0.95, reference_text, transform=axes[1, 1].transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        # Color coding for the result
        if result['is_healthy']:
            fig.patch.set_facecolor('lightgreen')
        else:
            fig.patch.set_facecolor('lightcoral')
        
        plt.tight_layout()
        
        # Save result if requested
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_prediction.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Result saved to: {output_path}")
        
        plt.show()
        
        return result
    
    def print_cure_information(self):
        """
        Print detailed cure information for Septoria brown spot
        """
        print("\n" + "="*60)
        print(self.cure_info['title'])
        print("="*60)
        
        print("\nTREATMENT METHODS:")
        for i, treatment in enumerate(self.cure_info['treatments'], 1):
            print(f"{i}. {treatment}")
        
        print("\nIMMEDIATE ACTIONS TO TAKE:")
        for action in self.cure_info['immediate_actions']:
            print(action)
        
        print(f"\nREFERENCE:")
        print(self.cure_info['reference'])
        print("="*60)
    
    def predict_with_cure_info(self, image_path):
        """
        Predict disease and show cure information if disease is detected
        """
        result, error = self.predict_image(image_path)
        
        if error:
            print(f"Error: {error}")
            return None
        
        print(f"\nPrediction Results for: {os.path.basename(image_path)}")
        print("-" * 50)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        if result['has_disease']:
            print("\n⚠️  DISEASE DETECTED: Septoria Brown Spot")
            self.print_cure_information()
        else:
            print("\n✅ HEALTHY LEAF - No disease detected")
            print("Continue regular monitoring and good plant health practices.")
        
        return result
    
    def batch_predict(self, image_directory, output_csv="predictions.csv"):
        """
        Predict disease for all images in a directory
        """
        if not os.path.exists(image_directory):
            print(f"Directory not found: {image_directory}")
            return None
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(image_directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print("No image files found in directory")
            return None
        
        print(f"Processing {len(image_files)} images...")
        
        # Process each image
        results = []
        successful = 0
        
        for i, filename in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {filename}")
            
            image_path = os.path.join(image_directory, filename)
            result, error = self.predict_image(image_path)
            
            if result:
                results.append({
                    'filename': filename,
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'is_healthy': result['is_healthy'],
                    'healthy_prob': result['probabilities']['healthy'],
                    'disease_prob': result['probabilities']['Septoria brown spot']
                })
                successful += 1
            else:
                print(f"  Error: {error}")
                results.append({
                    'filename': filename,
                    'predicted_class': 'ERROR',
                    'confidence': 0,
                    'is_healthy': False,
                    'healthy_prob': 0,
                    'disease_prob': 0
                })
        
        # Create DataFrame and save
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        print(f"\nBatch prediction completed!")
        print(f"Successful predictions: {successful}/{len(image_files)}")
        print(f"Results saved to: {output_csv}")
        
        # Summary statistics
        if successful > 0:
            healthy_count = df[df['is_healthy'] == True].shape[0]
            diseased_count = df[df['is_healthy'] == False].shape[0]
            
            print(f"\nPrediction Summary:")
            print(f"Healthy leaves: {healthy_count}")
            print(f"Diseased leaves: {diseased_count}")
            print(f"Average confidence: {df['confidence'].mean():.1%}")
            
            if diseased_count > 0:
                print(f"\n⚠️  {diseased_count} diseased leaves detected!")
                print("Please refer to the cure information for Septoria brown spot treatment.")
        
        return df

def interactive_prediction():
    """
    Interactive prediction interface
    """
    print("SOYBEAN DISEASE PREDICTION INTERFACE")
    print("=" * 50)
    
    try:
        predictor = SoybeanDiseasePredictor()
    except:
        print("Could not load model. Please train the model first.")
        return
    
    while True:
        print("\nChoose an option:")
        print("1. Predict single image")
        print("2. Predict single image with visualization")
        print("3. Predict with cure information (text)")
        print("4. Show cure information for Septoria brown spot")
        print("5. Batch predict directory")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip('"')
            result, error = predictor.predict_image(image_path)
            
            if result:
                print(f"\nPrediction: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"Health Status: {'Healthy' if result['is_healthy'] else 'Diseased'}")
            else:
                print(f"Error: {error}")
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip('"')
            save = input("Save visualization? (y/n): ").strip().lower() == 'y'
            predictor.predict_with_visualization(image_path, save_result=save)
        
        elif choice == '3':
            image_path = input("Enter image path: ").strip('"')
            predictor.predict_with_cure_info(image_path)
        
        elif choice == '4':
            predictor.print_cure_information()
        
        elif choice == '5':
            directory = input("Enter directory path: ").strip('"')
            output_file = input("Enter output CSV name (default: predictions.csv): ").strip()
            if not output_file:
                output_file = "predictions.csv"
            predictor.batch_predict(directory, output_file)
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

def quick_test_predictions():
    """
    Quick test function to verify model works
    """
    print("Quick Model Test")
    print("-" * 30)
    
    # Test on sample images from dataset
    dataset_base = r"C:\Users\Parth\OneDrive\Desktop\Soybean_Dataset_SuperSafe"
    
    predictor = SoybeanDiseasePredictor()
    
    # Test healthy image
    healthy_dir = os.path.join(dataset_base, "healthy")
    if os.path.exists(healthy_dir):
        healthy_files = [f for f in os.listdir(healthy_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if healthy_files:
            test_healthy = os.path.join(healthy_dir, healthy_files[0])
            print(f"Testing healthy image: {healthy_files[0]}")
            result = predictor.predict_with_visualization(test_healthy)
    
    # Test diseased image
    diseased_dir = os.path.join(dataset_base, "Septoria brown spot")
    if os.path.exists(diseased_dir):
        diseased_files = [f for f in os.listdir(diseased_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if diseased_files:
            test_diseased = os.path.join(diseased_dir, diseased_files[0])
            print(f"\nTesting diseased image: {diseased_files[0]}")
            result = predictor.predict_with_visualization(test_diseased)

if __name__ == "__main__":
    print("1. Interactive prediction interface")
    print("2. Quick test on sample images")
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        interactive_prediction()
    elif choice == "2":
        quick_test_predictions()
    else:
        print("Invalid choice")