#!/usr/bin/env python3
"""
Simple Fingerprint Pattern Classification GUI
A clean, easy-to-use interface for fingerprint pattern prediction
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os
import sys

class SimpleFingerprint:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Pattern Classifier")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.current_image = None
        self.image_path = None
        
        # Load model
        self.load_model()
        
        # Setup GUI
        self.setup_gui()
        
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = os.path.join('model', 'improved_pattern_cnn_model_retrained.h5')
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return False
                
            self.model = tf.keras.models.load_model(model_path)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return False
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, text="Fingerprint Pattern Classifier", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack()
        
        # Upload button
        upload_frame = tk.Frame(self.root, bg='#f0f0f0')
        upload_frame.pack(pady=20)
        
        self.upload_btn = tk.Button(upload_frame, text="📁 Select Fingerprint Image", 
                                   font=('Arial', 12), bg='#4CAF50', fg='white',
                                   command=self.select_image, padx=20, pady=10)
        self.upload_btn.pack()
        
        # Image display frame
        self.image_frame = tk.Frame(self.root, bg='white', relief='sunken', bd=2)
        self.image_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        self.image_label = tk.Label(self.image_frame, text="No image selected", 
                                   bg='white', font=('Arial', 12))
        self.image_label.pack(expand=True)
        
        # Results frame
        self.result_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.result_frame.pack(pady=10, padx=20, fill='x')
        
        # Prediction display
        self.prediction_label = tk.Label(self.result_frame, text="", 
                                        font=('Arial', 14, 'bold'), bg='#f0f0f0')
        self.prediction_label.pack(pady=5)
        
        # Confidence display
        self.confidence_label = tk.Label(self.result_frame, text="", 
                                        font=('Arial', 10), bg='#f0f0f0')
        self.confidence_label.pack()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select an image to classify")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief='sunken', anchor='w', bg='#e0e0e0')
        status_bar.pack(side='bottom', fill='x')
        
    def select_image(self):
        """Open file dialog to select an image"""
        filetypes = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff *.gif'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title='Select Fingerprint Image',
            filetypes=filetypes
        )
        
        if filename:
            self.image_path = filename
            self.load_and_display_image(filename)
            self.predict_pattern()
    
    def load_and_display_image(self, image_path):
        """Load and display the selected image"""
        try:
            # Load image
            image = Image.open(image_path)
            self.current_image = image.copy()
            
            # Resize for display
            display_size = (300, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Display image
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            self.status_var.set(f"Image loaded: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image")
    
    def predict_pattern(self):
        """Predict the fingerprint pattern"""
        if self.model is None or self.current_image is None:
            return
            
        try:
            # Preprocess image
            img_array = self.preprocess_image(self.current_image)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get class names
            class_names = ['Arc', 'Whorl', 'Loop']
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            # Update display
            self.prediction_label.configure(text=f"Pattern: {predicted_class}")
            self.confidence_label.configure(text=f"Confidence: {confidence:.1f}%")
            
            # Color code the prediction based on confidence
            if confidence >= 90:
                color = '#4CAF50'  # Green
            elif confidence >= 75:
                color = '#FF9800'  # Orange
            else:
                color = '#f44336'  # Red
                
            self.prediction_label.configure(fg=color)
            
            self.status_var.set(f"Prediction: {predicted_class} ({confidence:.1f}%)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to model input size
        image = image.resize((128, 128), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension and channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Channel dimension
        
        return img_array

def main():
    """Main function"""
    # Check if model exists
    if not os.path.exists('model/improved_pattern_cnn_model_retrained.h5'):
        print("❌ Error: Model file not found!")
        print("Please ensure the model file exists at: model/improved_pattern_cnn_model_retrained.h5")
        return
    
    # Create and run GUI
    root = tk.Tk()
    app = SimpleFingerprint(root)
    
    print("🚀 Starting Simple Fingerprint GUI...")
    print("📁 Select an image to classify fingerprint patterns")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Application closed")

if __name__ == "__main__":
    main()