import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DigitalPlantationTracker:
    def __init__(self):
        # Initialize models
        self.rf_growth_model = None
        self.rf_health_model = None
        self.kmeans_model = None
        self.kmeans_scaler = None
        self.cnn_model = None
        
        # Create sample datasets
        self.create_sample_datasets()
        
        # Train models
        self.train_models()
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Digital Plantation Tracker")
        self.root.geometry("1000x800")
        self.setup_gui()
        
    def create_sample_datasets(self):
        """Create sample datasets for demonstration"""
        # Tree growth and health data
        num_samples = 500
        weather = np.random.uniform(10, 40, num_samples)
        rainfall = np.random.uniform(0, 300, num_samples)
        soil_ph = np.random.uniform(4.5, 8.5, num_samples)
        soil_nutrients = np.random.uniform(1, 10, num_samples)
        age = np.random.randint(1, 60, num_samples)
        
        # Simulate growth and health
        growth = 50 + 0.8*age + 0.5*soil_nutrients + 0.3*rainfall - 0.5*(weather-25)**2 + np.random.normal(0, 10, num_samples)
        health = 70 + 0.3*soil_nutrients - 5*abs(soil_ph-6.5) + 0.2*rainfall + np.random.normal(0, 8, num_samples)
        health = np.clip(health, 0, 100)
        
        self.tree_data = pd.DataFrame({
            'weather': weather,
            'rainfall': rainfall,
            'soil_ph': soil_ph,
            'soil_nutrients': soil_nutrients,
            'age': age,
            'growth': growth,
            'health': health
        })
        
        # Create sample image dataset (in memory)
        self.create_sample_image_dataset()
    
    def create_sample_image_dataset(self):
        """Create a small sample image dataset in memory with healthy and diseased trees"""
        # This is a simplified version - in practice, you'd use real images
        self.image_classes = ['not_tree', 'healthy_tree', 'diseased_tree']
        self.cnn_input_shape = (128, 128, 3)
        
        # Generate some simple images to simulate the dataset
        num_samples_per_class = 50
        
        # Healthy tree images (green with uniform color)
        healthy_images = []
        for _ in range(num_samples_per_class):
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            # Trunk
            cv2.rectangle(img, (50, 80), (78, 120), (50, 30, 10), -1)
            # Leaves - healthy green
            cv2.circle(img, (64, 50), 40, (0, np.random.randint(100, 200), 0), -1)
            healthy_images.append(img)
        
        # Diseased tree images (with spots/discoloration)
        diseased_images = []
        for _ in range(num_samples_per_class):
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            # Trunk
            cv2.rectangle(img, (50, 80), (78, 120), (50, 30, 10), -1)
            # Leaves - patchy/discolored
            cv2.circle(img, (64, 50), 40, (0, np.random.randint(50, 150), 0), -1)
            # Add disease spots
            for _ in range(np.random.randint(5, 15)):
                x, y = np.random.randint(20, 108, 2)
                color = (np.random.randint(100, 200), np.random.randint(0, 100), 0)
                cv2.circle(img, (x, y), np.random.randint(2, 8), color, -1)
            diseased_images.append(img)
        
        # Non-tree images (random patterns)
        not_tree_images = []
        for _ in range(num_samples_per_class):
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            not_tree_images.append(img)
        
        # Combine and label (0: not_tree, 1: healthy, 2: diseased)
        self.image_data = np.array(not_tree_images + healthy_images + diseased_images) / 255.0
        self.image_labels = np.array(
            [0]*num_samples_per_class +  # not_tree
            [1]*num_samples_per_class +  # healthy
            [2]*num_samples_per_class    # diseased
        )
        
        # Shuffle
        indices = np.arange(len(self.image_labels))
        np.random.shuffle(indices)
        self.image_data = self.image_data[indices]
        self.image_labels = self.image_labels[indices]
    
    def train_models(self):
        """Train all ML models"""
        self.train_random_forest()
        self.train_kmeans()
        self.train_cnn()
    
    def train_random_forest(self):
        """Train Random Forest models for growth and health prediction"""
        X = self.tree_data[['weather', 'rainfall', 'soil_ph', 'soil_nutrients', 'age']]
        y_growth = self.tree_data['growth']
        y_health = self.tree_data['health']
        
        # Split data
        X_train, X_test, y_train_growth, y_test_growth = train_test_split(X, y_growth, test_size=0.2)
        _, _, y_train_health, y_test_health = train_test_split(X, y_health, test_size=0.2)
        
        # Growth prediction model
        self.rf_growth_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_growth_model.fit(X_train, y_train_growth)
        
        # Health prediction model
        self.rf_health_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_health_model.fit(X_train, y_train_health)
    
    def train_kmeans(self):
        """Train K-Means clustering model for resource allocation"""
        features = self.tree_data[['weather', 'rainfall', 'soil_ph', 'soil_nutrients']]
        
        # Standardize features
        self.kmeans_scaler = StandardScaler()
        features_scaled = self.kmeans_scaler.fit_transform(features)
        
        # Perform K-Means
        self.kmeans_model = KMeans(n_clusters=3, random_state=42)
        self.kmeans_model.fit(features_scaled)
    
    def train_cnn(self):
        """Train CNN model for tree image validation and health classification"""
        # Split image data
        X_train, X_test, y_train, y_test = train_test_split(
            self.image_data, self.image_labels, test_size=0.2, random_state=42)
        
        # Convert labels to one-hot encoding
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)
        
        # Build model
        self.cnn_model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.cnn_input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')  # 3 classes: not_tree, healthy, diseased
        ])
        
        self.cnn_model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
        
        # Train model
        history = self.cnn_model.fit(X_train, y_train_onehot, epochs=15, 
                                   validation_data=(X_test, y_test_onehot),
                                   batch_size=32)
        
        # Plot training history (for debugging)
        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('CNN Training History')
        plt.legend()
        plt.show()
    
    def predict_growth_and_health(self, weather, rainfall, soil_ph, soil_nutrients, age):
        """Predict tree growth and health"""
        input_data = pd.DataFrame({
            'weather': [weather],
            'rainfall': [rainfall],
            'soil_ph': [soil_ph],
            'soil_nutrients': [soil_nutrients],
            'age': [age]
        })
        
        growth = self.rf_growth_model.predict(input_data)[0]
        health = self.rf_health_model.predict(input_data)[0]
        
        return growth, health
    
    def recommend_resource_allocation(self, weather, rainfall, soil_ph, soil_nutrients):
        """Recommend resource allocation"""
        input_data = pd.DataFrame({
            'weather': [weather],
            'rainfall': [rainfall],
            'soil_ph': [soil_ph],
            'soil_nutrients': [soil_nutrients]
        })
        
        input_scaled = self.kmeans_scaler.transform(input_data)
        cluster = self.kmeans_model.predict(input_scaled)[0]
        
        recommendations = [
            "High Priority - Needs immediate attention and resources",
            "Medium Priority - Schedule maintenance and monitoring",
            "Low Priority - Minimal intervention needed"
        ]
        
        return cluster, recommendations[cluster]
    
    def check_tree_health(self, image_path):
        """Check if an image is a valid tree image and assess health"""
        try:
            # Load and preprocess image
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.cnn_model.predict(img_array)[0]
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            
            # Get class label and confidence
            status_map = {
                0: ("INVALID (Not a tree)", "red"),
                1: ("HEALTHY TREE", "green"),
                2: ("DISEASED TREE", "orange")
            }
            
            status, color = status_map.get(predicted_class, ("UNKNOWN", "black"))
            return status, color, confidence, predictions
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return "ERROR PROCESSING IMAGE", "red", 0.0, None
    
    def setup_gui(self):
        """Set up the graphical user interface"""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.tab_prediction = ttk.Frame(self.notebook)
        self.tab_allocation = ttk.Frame(self.notebook)
        self.tab_image = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_prediction, text="Growth/Health Prediction")
        self.notebook.add(self.tab_allocation, text="Resource Allocation")
        self.notebook.add(self.tab_image, text="Tree Health Detection")
        
        # Tab 1: Growth/Health Prediction
        self.setup_prediction_tab()
        
        # Tab 2: Resource Allocation
        self.setup_allocation_tab()
        
        # Tab 3: Image Validation
        self.setup_image_tab()
    
    def setup_prediction_tab(self):
        """Set up the prediction tab"""
        # Input frame
        input_frame = ttk.LabelFrame(self.tab_prediction, text="Input Parameters", padding=10)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        # Weather
        ttk.Label(input_frame, text="Temperature (°C):").grid(row=0, column=0, sticky='w')
        self.weather_entry = ttk.Entry(input_frame)
        self.weather_entry.grid(row=0, column=1, padx=5, pady=5)
        self.weather_entry.insert(0, "25.0")
        
        # Rainfall
        ttk.Label(input_frame, text="Rainfall (mm):").grid(row=1, column=0, sticky='w')
        self.rainfall_entry = ttk.Entry(input_frame)
        self.rainfall_entry.grid(row=1, column=1, padx=5, pady=5)
        self.rainfall_entry.insert(0, "150.0")
        
        # Soil pH
        ttk.Label(input_frame, text="Soil pH:").grid(row=2, column=0, sticky='w')
        self.soil_ph_entry = ttk.Entry(input_frame)
        self.soil_ph_entry.grid(row=2, column=1, padx=5, pady=5)
        self.soil_ph_entry.insert(0, "6.5")
        
        # Soil Nutrients
        ttk.Label(input_frame, text="Soil Nutrients (1-10):").grid(row=3, column=0, sticky='w')
        self.soil_nutrients_entry = ttk.Entry(input_frame)
        self.soil_nutrients_entry.grid(row=3, column=1, padx=5, pady=5)
        self.soil_nutrients_entry.insert(0, "7.0")
        
        # Age
        ttk.Label(input_frame, text="Tree Age (months):").grid(row=4, column=0, sticky='w')
        self.age_entry = ttk.Entry(input_frame)
        self.age_entry.grid(row=4, column=1, padx=5, pady=5)
        self.age_entry.insert(0, "24")
        
        # Predict button
        predict_btn = ttk.Button(input_frame, text="Predict", command=self.run_prediction)
        predict_btn.grid(row=5, columnspan=2, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.tab_prediction, text="Prediction Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.growth_label = ttk.Label(results_frame, text="Predicted Growth: ")
        self.growth_label.pack(anchor='w', pady=5)
        
        self.health_label = ttk.Label(results_frame, text="Predicted Health: ")
        self.health_label.pack(anchor='w', pady=5)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Growth vs. Health Relationship")
        ax.scatter(self.tree_data['growth'], self.tree_data['health'], alpha=0.5)
        ax.set_xlabel("Growth (cm)")
        ax.set_ylabel("Health Score")
        
        self.canvas_pred = FigureCanvasTkAgg(fig, master=results_frame)
        self.canvas_pred.draw()
        self.canvas_pred.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_allocation_tab(self):
        """Set up the resource allocation tab"""
        # Input frame
        input_frame = ttk.LabelFrame(self.tab_allocation, text="Area Parameters", padding=10)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        # Weather
        ttk.Label(input_frame, text="Temperature (°C):").grid(row=0, column=0, sticky='w')
        self.alloc_weather_entry = ttk.Entry(input_frame)
        self.alloc_weather_entry.grid(row=0, column=1, padx=5, pady=5)
        self.alloc_weather_entry.insert(0, "28.0")
        
        # Rainfall
        ttk.Label(input_frame, text="Rainfall (mm):").grid(row=1, column=0, sticky='w')
        self.alloc_rainfall_entry = ttk.Entry(input_frame)
        self.alloc_rainfall_entry.grid(row=1, column=1, padx=5, pady=5)
        self.alloc_rainfall_entry.insert(0, "120.0")
        
        # Soil pH
        ttk.Label(input_frame, text="Soil pH:").grid(row=2, column=0, sticky='w')
        self.alloc_soil_ph_entry = ttk.Entry(input_frame)
        self.alloc_soil_ph_entry.grid(row=2, column=1, padx=5, pady=5)
        self.alloc_soil_ph_entry.insert(0, "6.2")
        
        # Soil Nutrients
        ttk.Label(input_frame, text="Soil Nutrients (1-10):").grid(row=3, column=0, sticky='w')
        self.alloc_soil_nutrients_entry = ttk.Entry(input_frame)
        self.alloc_soil_nutrients_entry.grid(row=3, column=1, padx=5, pady=5)
        self.alloc_soil_nutrients_entry.insert(0, "6.5")
        
        # Recommend button
        recommend_btn = ttk.Button(input_frame, text="Get Recommendation", command=self.run_allocation)
        recommend_btn.grid(row=4, columnspan=2, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.tab_allocation, text="Allocation Recommendation", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.cluster_label = ttk.Label(results_frame, text="Assigned Cluster: ")
        self.cluster_label.pack(anchor='w', pady=5)
        
        self.recommendation_label = ttk.Label(results_frame, text="Recommendation: ")
        self.recommendation_label.pack(anchor='w', pady=5)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Plantation Areas Clustering")
        
        # Get cluster assignments for all data points
        features = self.tree_data[['weather', 'rainfall', 'soil_ph', 'soil_nutrients']]
        features_scaled = self.kmeans_scaler.transform(features)
        clusters = self.kmeans_model.predict(features_scaled)
        
        ax.scatter(features['soil_ph'], features['soil_nutrients'], c=clusters, cmap='viridis')
        ax.set_xlabel("Soil pH")
        ax.set_ylabel("Soil Nutrients")
        
        self.canvas_alloc = FigureCanvasTkAgg(fig, master=results_frame)
        self.canvas_alloc.draw()
        self.canvas_alloc.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_image_tab(self):
        """Set up the image health detection tab"""
        # Image frame
        image_frame = ttk.LabelFrame(self.tab_image, text="Tree Health Detection", padding=10)
        image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Select image button
        select_btn = ttk.Button(image_frame, text="Select Image", command=self.select_image)
        select_btn.pack(pady=10)
        
        # Image display
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack()
        
        # Results
        self.image_result_label = ttk.Label(
            image_frame, 
            text="", 
            font=('Helvetica', 14, 'bold'),
            justify='center'
        )
        self.image_result_label.pack(pady=10)
        
        # Confidence
        self.confidence_label = ttk.Label(
            image_frame, 
            text="",
            font=('Helvetica', 12)
        )
        self.confidence_label.pack()
        
        # Class probabilities
        self.probs_frame = ttk.Frame(image_frame)
        self.probs_frame.pack(pady=10)
        
        ttk.Label(self.probs_frame, text="Classification Probabilities:", font=('Helvetica', 10, 'bold')).grid(row=0, columnspan=3)
        
        self.not_tree_prob = ttk.Label(self.probs_frame, text="Not Tree: 0%")
        self.not_tree_prob.grid(row=1, column=0, padx=5)
        
        self.healthy_prob = ttk.Label(self.probs_frame, text="Healthy: 0%")
        self.healthy_prob.grid(row=1, column=1, padx=5)
        
        self.diseased_prob = ttk.Label(self.probs_frame, text="Diseased: 0%")
        self.diseased_prob.grid(row=1, column=2, padx=5)
    
    def run_prediction(self):
        """Run growth and health prediction"""
        try:
            weather = float(self.weather_entry.get())
            rainfall = float(self.rainfall_entry.get())
            soil_ph = float(self.soil_ph_entry.get())
            soil_nutrients = float(self.soil_nutrients_entry.get())
            age = int(self.age_entry.get())
            
            growth, health = self.predict_growth_and_health(weather, rainfall, soil_ph, soil_nutrients, age)
            
            self.growth_label.config(text=f"Predicted Growth: {growth:.1f} cm")
            self.health_label.config(text=f"Predicted Health: {health:.1f}/100")
            
            # Update visualization with new point
            fig = self.canvas_pred.figure
            ax = fig.axes[0]
            ax.plot(growth, health, 'ro', markersize=10)  # Add new point in red
            self.canvas_pred.draw()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
    
    def run_allocation(self):
        """Run resource allocation recommendation"""
        try:
            weather = float(self.alloc_weather_entry.get())
            rainfall = float(self.alloc_rainfall_entry.get())
            soil_ph = float(self.alloc_soil_ph_entry.get())
            soil_nutrients = float(self.alloc_soil_nutrients_entry.get())
            
            cluster, recommendation = self.recommend_resource_allocation(
                weather, rainfall, soil_ph, soil_nutrients)
            
            self.cluster_label.config(text=f"Assigned Cluster: {cluster}")
            self.recommendation_label.config(text=f"Recommendation: {recommendation}")
            
            # Update visualization with new point
            fig = self.canvas_alloc.figure
            ax = fig.axes[0]
            
            # Plot the new point
            ax.plot(soil_ph, soil_nutrients, 'ro', markersize=10)
            self.canvas_alloc.draw()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
    
    def select_image(self):
        """Select and validate an image"""
        file_path = filedialog.askopenfilename(
            title="Select Tree Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Display image
                img = Image.open(file_path)
                img.thumbnail((400, 400))
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk
                
                # Check tree health
                status, color, confidence, probs = self.check_tree_health(file_path)
                
                # Update status
                self.image_result_label.config(text=status, foreground=color)
                
                # Update confidence
                self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
                
                # Update probabilities if available
                if probs is not None:
                    self.not_tree_prob.config(text=f"Not Tree: {probs[0]*100:.1f}%")
                    self.healthy_prob.config(text=f"Healthy: {probs[1]*100:.1f}%")
                    self.diseased_prob.config(text=f"Diseased: {probs[2]*100:.1f}%")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")

# Run the application
if __name__ == "__main__":
    try:
        app = DigitalPlantationTracker()
        app.root.mainloop()
    except ImportError as e:
        print(f"Error: {e}. Please make sure you have all required packages installed.")
        print("You can install them with: pip install numpy pandas scikit-learn tensorflow pillow matplotlib tk")