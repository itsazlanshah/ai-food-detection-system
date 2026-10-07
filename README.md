# AI Food Detection & Classification System

An AI-powered food detection and nutritional analysis system developed as a university Artificial Intelligence semester project.

**Live Demo:** Not deployed yet

## Overview

This application uses computer vision and deep learning to identify food items from images. After detecting a food item, the system provides nutritional information such as calories, protein, carbohydrates, fats, fiber, and sugar, along with a health score and dietary tips.

## Features

* **Food Detection:** Identifies 20+ different food categories using YOLOv11.
* **Nutritional Information:** Provides calories, protein, carbohydrates, fats, fiber, and sugar.
* **Portion Control:** Allows users to adjust portion size in grams and calculate nutritional values accordingly.
* **Image & Camera Input:** Supports both image uploads and direct camera capture.
* **Health Score:** Provides a health score from 1–10 with dietary recommendations.
* **Interactive Interface:** Built using Streamlit for a simple and interactive user experience.

## Technologies Used

* **Python** — Core programming language
* **YOLOv11 (Ultralytics)** — Food detection and classification
* **PyTorch** — Deep learning model inference
* **Streamlit** — Web interface
* **OpenCV & PIL** — Image processing and camera integration
* **Pandas & NumPy** — Data processing and nutritional calculations

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai-food-detection-system
```

### 2. Prepare the Model

Place the trained model file `best.pt` in the root directory of the project.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install ultralytics torch
```

### 4. Run the Application

```bash
streamlit run "app (1).py"
```

The application will open in your browser.

## Project Team

**Azlan Shah**
**Muhammad Nofal Zia**

This project was developed collaboratively as part of our Artificial Intelligence semester project at Bahria University.

## Purpose

The project was developed to explore the practical application of deep learning and computer vision in health and nutrition. It provided hands-on experience with object detection, model inference, image processing, and building an interactive AI application.

## Future Improvements

* Deploy the application online.
* Expand the food dataset and number of supported food categories.
* Improve detection accuracy for multiple food items in a single image.
* Add user profiles and meal history.
* Provide more personalized nutritional recommendations.
