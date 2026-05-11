# Artificial Intelegence Semester Project

**Live demo:** Not deployed yet

## What it does
This application is an **AI-powered Food Nutrition Classifier** that uses computer vision to identify various food items from images. Once identified, it provides comprehensive nutritional data, including calories, protein, carbs, fats, and personalized health tips, helping users track their dietary intake with ease.

## Tech stack
- **Python**: Core programming language.
- **Streamlit**: Framework for building the interactive web interface.
- **YOLOv11 (Ultralytics)**: State-of-the-art object detection and classification model.
- **PyTorch**: Backend engine for deep learning model inference.
- **OpenCV & PIL**: Used for image preprocessing and camera integration.
- **Pandas & NumPy**: For efficient data handling and nutritional calculations.

## Key features
- **Real-time Identification**: Recognizes 20+ different food categories (from Chicken Curry to Steak) with high confidence.
- **Detailed Nutrition Profiles**: Displays complete macronutrient breakdowns (Protein, Carbs, Fat, Fiber, Sugar) for every detected item.
- **Dynamic Portion Control**: Allows users to adjust the portion size (in grams) to scale nutritional values accurately for their specific meal.
- **Dual Capture Methods**: Support for both local image uploads and direct camera capture within the web app.
- **Health Scoring System**: Provides an AI-calculated health score (1-10) and actionable dietary tips for each food item.

## Getting started
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-food-detection-system
   ```

2. **Prepare the Model**:
   The application expects the model file to be in the root directory. Copy the weights:
   ```bash
   # On Windows
   copy yolov11m_cls_50epochs45\weights\best.pt .\best.pt
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # Note: Ensure ultralytics and torch are also installed
   pip install ultralytics torch
   ```

4. **Launch the App**:
   ```bash
   streamlit run "app (1).py"
   ```

## Why I built this
I developed this project for my Artificial Intelligence class at university to explore how deep learning can be applied to real-world health and wellness challenges. It served as a successful demonstration of real-time computer vision during my semester project demos.
