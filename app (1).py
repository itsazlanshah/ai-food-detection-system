"""
🍔 AI FOOD NUTRITION CLASSIFIER - IMPROVED VERSION
Fixed UI, static calories per 100g, better layout
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import cv2
import torch
import os

# Page config
st.set_page_config(page_title="Food Nutrition AI", page_icon="🍔", layout="wide")

# Enhanced CSS with fixed button sizes
st.markdown("""
<style>
    .main-title {text-align: center; color: #667eea; font-size: 3.5rem; font-weight: bold; margin-bottom: 10px;}
    .sub-title {text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 40px;}
    
    .food-card {
        background: white; 
        border-radius: 20px; 
        padding: 25px; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.08); 
        transition: all 0.3s; 
        border: 2px solid transparent; 
        margin-bottom: 20px;
        height: 100%;
    }
    .food-card:hover {
        transform: translateY(-8px); 
        box-shadow: 0 15px 50px rgba(102,126,234,0.25); 
        border: 2px solid #667eea;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
        border-radius: 30px; 
        padding: 40px; 
        color: white; 
        text-align: center; 
        box-shadow: 0 20px 60px rgba(245,87,108,0.4);
        margin: 20px 0;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px 20px;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: bold;
        border: none;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(102,126,234,0.4);
        transition: all 0.3s ease;
        text-align: center;
        display: block;
        width: 100%;
        margin: 20px 0;
    }
    .cta-button:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102,126,234,0.6);
    }
    
    .nutrition-badge {
        display: inline-block; 
        padding: 12px 20px; 
        border-radius: 25px; 
        margin: 8px 5px; 
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    }
    
    .calories-low {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white;}
    .calories-medium {background: linear-gradient(135deg, #FFA94D 0%, #FF922B 100%); color: white;}
    .calories-high {background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); color: white;}
    .protein-badge {background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); color: white;}
    .carbs-badge {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;}
    .fat-badge {background: linear-gradient(135deg, #fad961 0%, #f76b1c 100%); color: white;}
    .fiber-badge {background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333;}
    
    .metric-box {
        background: white; 
        border-radius: 15px; 
        padding: 20px; 
        box-shadow: 0 5px 20px rgba(0,0,0,0.08); 
        text-align: center;
        margin: 10px 0;
    }
    .metric-box h3 {color: #667eea; font-size: 2rem; margin: 10px 0;}
    
    .tip-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
        border-radius: 15px; 
        padding: 20px; 
        border-left: 5px solid #ff6b6b; 
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); 
        border-radius: 15px; 
        padding: 20px; 
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    /* Fix sidebar button sizes */
    .stButton button {
        width: 100%;
        height: 45px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        margin: 3px 0;
    }
    
    /* Specific styling for the main CTA button */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
        height: 80px !important;
        border: none !important;
        border-radius: 15px !important;
        box-shadow: 0 10px 30px rgba(102,126,234,0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin: 40px 0 !important;
        padding: 20px !important;
    }
    
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 15px 40px rgba(102,126,234,0.6) !important;
    }
    
    /* Style for the sidebar About button */
    div[data-testid="stSidebar"] button:last-child {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        height: 50px !important;
        font-size: 1.1rem !important;
        margin-top: 20px !important;
    }
    
    /* Center align the main button container */
    .main-cta-container {
        text-align: center;
        margin: 40px 0;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'uploaded_image' not in st.session_state: st.session_state.uploaded_image = None
if 'results' not in st.session_state: st.session_state.results = None
if 'model' not in st.session_state: st.session_state.model = None
if 'class_names' not in st.session_state: st.session_state.class_names = None
if 'portion_size' not in st.session_state: st.session_state.portion_size = 100
if 'confidence_threshold' not in st.session_state: st.session_state.confidence_threshold = 0.3

# STATIC NUTRITION DATABASE (per 100g) - Matches your 20 classes exactly
FOOD_DB = {
    'chicken_curry': {'name': 'Chicken Curry', 'emoji': '🍛', 'calories': 185, 'protein': 18, 'carbs': 12, 'fat': 8, 'fiber': 2, 'sugar': 3, 'sodium': 400, 'serving': '1 cup (250g)', 'category': 'Main Dish', 'tip': 'Rich in protein and anti-inflammatory spices. Great source of vitamins!', 'health_score': 8},
    'chicken_wings': {'name': 'Chicken Wings', 'emoji': '🍗', 'calories': 290, 'protein': 27, 'carbs': 0, 'fat': 20, 'fiber': 0, 'sugar': 0, 'sodium': 600, 'serving': '4 wings (100g)', 'category': 'Protein', 'tip': 'High protein but also high in fat. Choose grilled over fried!', 'health_score': 6},
    'chocolate_cake': {'name': 'Chocolate Cake', 'emoji': '🍰', 'calories': 370, 'protein': 5, 'carbs': 50, 'fat': 18, 'fiber': 2, 'sugar': 35, 'sodium': 300, 'serving': '1 slice (100g)', 'category': 'Dessert', 'tip': 'High in sugar and calories. Enjoy as an occasional treat only!', 'health_score': 3},
    'club_sandwich': {'name': 'Club Sandwich', 'emoji': '🥪', 'calories': 240, 'protein': 15, 'carbs': 28, 'fat': 9, 'fiber': 3, 'sugar': 4, 'sodium': 500, 'serving': '1 sandwich (200g)', 'category': 'Meal', 'tip': 'Balanced meal with protein, carbs and veggies. Use whole grain bread!', 'health_score': 7},
    'cup_cakes': {'name': 'Cupcakes', 'emoji': '🧁', 'calories': 305, 'protein': 3, 'carbs': 45, 'fat': 13, 'fiber': 1, 'sugar': 30, 'sodium': 250, 'serving': '1 cupcake (50g)', 'category': 'Dessert', 'tip': 'Sweet treat high in sugar. Save for special occasions!', 'health_score': 2},
    'donuts': {'name': 'Donuts', 'emoji': '🍩', 'calories': 452, 'protein': 5, 'carbs': 51, 'fat': 25, 'fiber': 2, 'sugar': 22, 'sodium': 300, 'serving': '1 donut (75g)', 'category': 'Dessert', 'tip': 'Very high in calories, sugar and fat. Opt for baked versions!', 'health_score': 2},
    'dumplings': {'name': 'Dumplings', 'emoji': '🥟', 'calories': 210, 'protein': 8, 'carbs': 28, 'fat': 7, 'fiber': 2, 'sugar': 2, 'sodium': 450, 'serving': '5 pieces (100g)', 'category': 'Main Dish', 'tip': 'Steamed dumplings are much healthier than fried!', 'health_score': 7},
    'fish_and_chips': {'name': 'Fish & Chips', 'emoji': '🐟', 'calories': 265, 'protein': 17, 'carbs': 25, 'fat': 11, 'fiber': 2, 'sugar': 1, 'sodium': 500, 'serving': '1 serving (200g)', 'category': 'Main Dish', 'tip': 'Good source of protein and omega-3 fatty acids. Watch portion size!', 'health_score': 6},
    'french_fries': {'name': 'French Fries', 'emoji': '🍟', 'calories': 312, 'protein': 4, 'carbs': 41, 'fat': 15, 'fiber': 4, 'sugar': 0, 'sodium': 300, 'serving': '1 medium (117g)', 'category': 'Side Dish', 'tip': 'High in calories and fat. Choose baked sweet potato fries instead!', 'health_score': 4},
    'french_toast': {'name': 'French Toast', 'emoji': '🍞', 'calories': 220, 'protein': 8, 'carbs': 30, 'fat': 7, 'fiber': 2, 'sugar': 8, 'sodium': 350, 'serving': '2 slices (100g)', 'category': 'Breakfast', 'tip': 'Use whole grain bread and limit syrup for healthier option!', 'health_score': 6},
    'fried_rice': {'name': 'Fried Rice', 'emoji': '🍚', 'calories': 228, 'protein': 5, 'carbs': 40, 'fat': 6, 'fiber': 1, 'sugar': 2, 'sodium': 600, 'serving': '1 cup (200g)', 'category': 'Main Dish', 'tip': 'Add more vegetables for extra nutrients. Use brown rice for more fiber!', 'health_score': 6},
    'grilled_cheese_sandwich': {'name': 'Grilled Cheese', 'emoji': '🧀', 'calories': 290, 'protein': 12, 'carbs': 28, 'fat': 15, 'fiber': 2, 'sugar': 3, 'sodium': 600, 'serving': '1 sandwich (100g)', 'category': 'Meal', 'tip': 'Good source of calcium. Add tomatoes for extra vitamins!', 'health_score': 6},
    'hamburger': {'name': 'Hamburger', 'emoji': '🍔', 'calories': 295, 'protein': 17, 'carbs': 24, 'fat': 14, 'fiber': 2, 'sugar': 5, 'sodium': 500, 'serving': '1 burger (150g)', 'category': 'Main Dish', 'tip': 'Choose lean beef and load up on vegetables. Skip the mayo!', 'health_score': 6},
    'hot_and_sour_soup': {'name': 'Hot & Sour Soup', 'emoji': '🍲', 'calories': 95, 'protein': 6, 'carbs': 10, 'fat': 3, 'fiber': 1, 'sugar': 2, 'sodium': 800, 'serving': '1 bowl (250ml)', 'category': 'Soup', 'tip': 'Low calorie and warming. Great for digestion but watch sodium content!', 'health_score': 7},
    'ice_cream': {'name': 'Ice Cream', 'emoji': '🍨', 'calories': 207, 'protein': 4, 'carbs': 24, 'fat': 11, 'fiber': 1, 'sugar': 21, 'sodium': 80, 'serving': '1/2 cup (100g)', 'category': 'Dessert', 'tip': 'Frozen treat high in sugar. Watch portion sizes and opt for lower-fat versions!', 'health_score': 4},
    'macaroni_and_cheese': {'name': 'Mac & Cheese', 'emoji': '🧈', 'calories': 164, 'protein': 7, 'carbs': 19, 'fat': 7, 'fiber': 1, 'sugar': 2, 'sodium': 450, 'serving': '1 cup (200g)', 'category': 'Main Dish', 'tip': 'Comfort food! Add vegetables and use whole grain pasta for better nutrition!', 'health_score': 5},
    'omelette': {'name': 'Omelette', 'emoji': '🍳', 'calories': 154, 'protein': 11, 'carbs': 1, 'fat': 12, 'fiber': 0, 'sugar': 1, 'sodium': 400, 'serving': '2 eggs (100g)', 'category': 'Breakfast', 'tip': 'Excellent source of protein! Add vegetables and use less butter for healthier version!', 'health_score': 8},
    'pizza': {'name': 'Pizza', 'emoji': '🍕', 'calories': 266, 'protein': 11, 'carbs': 33, 'fat': 10, 'fiber': 2, 'sugar': 4, 'sodium': 600, 'serving': '1 slice (100g)', 'category': 'Main Dish', 'tip': 'Choose thin crust and load up on veggie toppings. Go easy on cheese!', 'health_score': 5},
    'samosa': {'name': 'Samosa', 'emoji': '🥟', 'calories': 262, 'protein': 5, 'carbs': 24, 'fat': 17, 'fiber': 3, 'sugar': 2, 'sodium': 400, 'serving': '1 piece (100g)', 'category': 'Snack', 'tip': 'Deep fried snack. Baked version is much healthier. Enjoy occasionally!', 'health_score': 4},
    'steak': {'name': 'Steak', 'emoji': '🥩', 'calories': 271, 'protein': 26, 'carbs': 0, 'fat': 18, 'fiber': 0, 'sugar': 0, 'sodium': 60, 'serving': '1 steak (100g)', 'category': 'Protein', 'tip': 'Excellent source of protein and iron! Choose lean cuts and moderate portions!', 'health_score': 7}
}

@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ '{model_path}' not found!")
        return None, {i: k for i, k in enumerate(FOOD_DB.keys())}
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        class_names = model.names if hasattr(model, 'names') else {i: k for i, k in enumerate(FOOD_DB.keys())}
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        model(dummy, verbose=False)
        st.success(f"✅ Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}!")
        return model, class_names
    except Exception as e:
        st.error(f"Error: {e}")
        return None, {i: k for i, k in enumerate(FOOD_DB.keys())}

def classify_food(image):
    if st.session_state.model is None:
        return [], 0
    img = np.array(image) if isinstance(image, Image.Image) else image
    img = cv2.resize(img, (224, 224))
    start = time.time()
    results = st.session_state.model(img, verbose=False)
    t = time.time() - start
    preds = []
    for result in results:
        if hasattr(result, 'probs'):
            probs = result.probs.data.cpu().numpy()
            for idx in np.argsort(probs)[::-1][:10]:
                conf = float(probs[idx])
                if conf < st.session_state.confidence_threshold:
                    continue
                key = st.session_state.class_names.get(int(idx))
                if key not in FOOD_DB:
                    continue
                food = FOOD_DB[key]
                mult = st.session_state.portion_size / 100
                preds.append({
                    'name': food['name'], 'emoji': food['emoji'], 'confidence': conf,
                    'calories': int(food['calories']*mult), 'protein': round(food['protein']*mult,1),
                    'carbs': round(food['carbs']*mult,1), 'fat': round(food['fat']*mult,1),
                    'fiber': round(food['fiber']*mult,1), 'sugar': round(food['sugar']*mult,1),
                    'sodium': int(food['sodium']*mult), 'serving': food['serving'],
                    'category': food['category'], 'tip': food['tip'], 'health_score': food['health_score']
                })
    return preds, t

def get_cal_level(cal):
    base = (cal / st.session_state.portion_size) * 100
    if base < 150: return 'Low', 'calories-low', '🟢'
    elif base < 300: return 'Medium', 'calories-medium', '🟡'
    else: return 'High', 'calories-high', '🔴'

# SIDEBAR
with st.sidebar:
    st.markdown("# 🍔 Food AI")
    st.markdown("---")
    
    st.markdown("### 🧭 Navigation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Home", use_container_width=True, key="btn_home"):
            st.session_state.page = 'home'
            st.rerun()
    with col2:
        if st.button("📸 Analyze", use_container_width=True, key="btn_analyze"):
            st.session_state.page = 'classify'
            st.rerun()
    
    # Add the "About" button below the Home and Analyze buttons
    if st.button("ℹ️ About", use_container_width=True, key="btn_about"):
        st.session_state.page = 'about'
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.session_state.portion_size = st.slider("Portion (g)", 50, 500, st.session_state.portion_size, 25)
    st.session_state.confidence_threshold = st.slider("Confidence", 0.0, 1.0, st.session_state.confidence_threshold, 0.05)
    
    st.markdown("---")
    st.markdown("### 🚦 Model Status")
    if st.session_state.model:
        st.success("✅ Ready")
        st.caption(f"**Device:** {'GPU 🔥' if torch.cuda.is_available() else 'CPU ⚡'}")
        st.caption(f"**Classes:** {len(st.session_state.class_names)}")
    else:
        st.error("❌ Not Loaded")
        if st.button("🔄 Load Model", use_container_width=True):
            st.session_state.model, st.session_state.class_names = load_model()
            st.rerun()
    
    if st.session_state.results:
        st.markdown("---")
        st.markdown("### 📈 Last Result")
        top = st.session_state.results['predictions'][0]
        st.caption(f"{top['emoji']} **{top['name']}**")
        st.caption(f"🔥 **{top['calories']} calories**")
        st.caption(f"✅ {top['confidence']:.0%} confidence")

# HOME PAGE
if st.session_state.page == 'home':
    st.markdown("<h1 class='main-title'>🍔 AI Food Nutrition Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Instantly identify food and discover complete nutritional information using AI</p>", unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.session_state.model, st.session_state.class_names = load_model()
    
    # REMOVED the 4 blue cards and replaced with just the CTA button
    
    # CTA Button - BIG and CENTERED
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if st.button("🚀 START ANALYZING FOOD NOW", use_container_width=True, type="primary", key="cta_button"):
            st.session_state.page = 'classify'
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Featured Foods - MOVED DOWN
    st.markdown("### 🌟 Featured Foods in Our Database")
    st.markdown("<br>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    for idx, (key, food) in enumerate(list(FOOD_DB.items())[:12]):
        with cols[idx % 4]:
            cal_lv, box_cls, emoji_ind = get_cal_level(food['calories'])
            st.markdown(f"""
            <div class='food-card'>
                <div style='text-align: center; font-size: 3.5rem; margin: 10px 0;'>{food['emoji']}</div>
                <h4 style='text-align: center; color: #2d3748; margin: 10px 0;'>{food['name']}</h4>
                <div class='nutrition-badge {box_cls}' style='width: 90%; margin: 10px auto; text-align: center;'>
                    {food['calories']} cal
                </div>
                <p style='text-align: center; font-size: 0.85rem; color: #718096; margin: 10px 0;'>
                    P: {food['protein']}g | C: {food['carbs']}g<br>F: {food['fat']}g | Fiber: {food['fiber']}g
                </p>
                <p style='text-align: center; font-size: 0.8rem; color: #a0aec0; margin-top: 8px;'>
                    {emoji_ind} {cal_lv} Calorie
                </p>
            </div>
            """, unsafe_allow_html=True)

# CLASSIFY PAGE
elif st.session_state.page == 'classify':
    st.markdown("# 📸 Food Nutrition Analysis")
    st.markdown("Upload or capture a photo of your food to get instant nutritional information")
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Model not loaded. Loading now...")
        st.session_state.model, st.session_state.class_names = load_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Upload Food Image")
        tab1, tab2, tab3 = st.tabs(["📁 Upload", "📷 Camera", "🎨 Sample"])
        
        with tab1:
            file = st.file_uploader("Choose image", type=['jpg','jpeg','png'])
            if file:
                img = Image.open(file)
                st.session_state.uploaded_image = img
                st.image(img, use_column_width=True)
        
        with tab2:
            cam = st.camera_input("Take picture")
            if cam:
                img = Image.open(cam)
                st.session_state.uploaded_image = img
                st.image(img, use_column_width=True)
        
        with tab3:
            samples = list(FOOD_DB.items())[:10]
            choice = st.selectbox("Choose:", [f"{f['emoji']} {f['name']}" for _,f in samples])
            if st.button("Generate Sample", use_container_width=True):
                img_arr = np.ones((400,400,3), dtype=np.uint8)*240
                cv2.circle(img_arr, (200,200), 140, (100,150,255), -1)
                st.session_state.uploaded_image = Image.fromarray(img_arr)
                st.image(st.session_state.uploaded_image, use_column_width=True)
    
    with col2:
        st.markdown("### 🤖 AI Analysis Results")
        if st.session_state.model:
            st.markdown(f"""
            <div class='info-box'>
                <strong>🤖 Model:</strong> YOLOv11m Classification<br>
                <strong>📏 Portion:</strong> {st.session_state.portion_size}g<br>
                <strong>🎯 Confidence Threshold:</strong> {st.session_state.confidence_threshold:.0%}
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.uploaded_image and st.session_state.model:
            if st.button("🔍 ANALYZE FOOD NOW", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing your food..."):
                    preds, t = classify_food(st.session_state.uploaded_image)
                    if preds:
                        st.session_state.results = {'predictions': preds, 'time': t}
                        st.success(f"✅ Analysis complete in {t:.2f} seconds!")
                    else:
                        st.warning("⚠️ No predictions above confidence threshold. Try adjusting settings in sidebar.")
        elif not st.session_state.uploaded_image:
            st.info("👆 Please upload or capture a food image first")
    
    if st.session_state.results and st.session_state.results.get('predictions'):
        st.markdown("---")
        top = st.session_state.results['predictions'][0]
        cal_lv, box_cls, emoji_ind = get_cal_level(top['calories'])
        
        st.markdown(f"""
        <div class='prediction-card'>
            <div style='font-size: 5rem; margin-bottom: 10px;'>{top['emoji']}</div>
            <h2 style='font-size: 2.5rem; margin: 10px 0;'>{top['name']}</h2>
            <h3 style='font-size: 2rem; margin: 15px 0;'>🔥 {top['calories']} Calories</h3>
            <p style='font-size: 1.2rem; margin: 5px 0;'>✅ {top['confidence']:.1%} Confidence</p>
            <p style='font-size: 1rem; opacity: 0.9;'>📏 Based on {st.session_state.portion_size}g portion</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown("### 📊 Nutrition Facts")
            st.markdown(f"""
            <div class='metric-box'>
                <h3>{top['protein']}g</h3>
                <p><span class='nutrition-badge protein-badge'>Protein</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='metric-box'>
                <h3>{top['carbs']}g</h3>
                <p><span class='nutrition-badge carbs-badge'>Carbs</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='metric-box'>
                <h3>{top['fat']}g</h3>
                <p><span class='nutrition-badge fat-badge'>Fat</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='metric-box'>
                <h3>{top['fiber']}g</h3>
                <p><span class='nutrition-badge fiber-badge'>Fiber</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r:
            st.markdown("### 🏷️ Food Details")
            st.markdown(f"""
            <div style='background: white; padding: 25px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05); border: 1px solid #e0e0e0;'>
                <table style='width: 100%; border-collapse: collapse; color: #333;'>
                    <tr>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: 600; color: #555;'>Category</td>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0; color: #222; font-weight: 500;'>{top['category']}</td>
                    </tr>
                    <tr>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: 600; color: #555;'>Serving Size</td>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0; color: #222; font-weight: 500;'>{top['serving']}</td>
                    </tr>
                    <tr>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: 600; color: #555;'>Sugar</td>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0; color: #222; font-weight: 500;'>{top['sugar']}g</td>
                    </tr>
                    <tr>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: 600; color: #555;'>Sodium</td>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0; color: #222; font-weight: 500;'>{top['sodium']}mg</td>
                    </tr>
                    <tr>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: 600; color: #555;'>Calorie Level</td>
                        <td style='padding: 12px; border-bottom: 1px solid #e0e0e0;'>
                            <span class='nutrition-badge {box_cls}' style='display: inline-block; padding: 6px 12px; border-radius: 12px; font-size: 0.85rem;'>
                                {cal_lv} ({top['calories']} cal)
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td style='padding: 12px; font-weight: 600; color: #555;'>Health Score</td>
                        <td style='padding: 12px; color: #222; font-weight: 500; font-size: 1.2rem;'>
                            {'⭐' * top['health_score']}
                            <span style='color: #777; font-size: 0.9rem; margin-left: 8px;'>
                                ({top['health_score']}/10)
                            </span>
                        </td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='tip-box'>
                <h4>💡 Nutrition Tip</h4>
                <p>{top['tip']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional predictions if any
        if len(st.session_state.results['predictions']) > 1:
            st.markdown("---")
            st.markdown("### 🔍 Other Possible Matches")
            cols = st.columns(min(4, len(st.session_state.results['predictions'])))
            for idx, pred in enumerate(st.session_state.results['predictions'][1:5]):
                with cols[idx % len(cols)]:
                    cal_lv, box_cls, emoji_ind = get_cal_level(pred['calories'])
                    st.markdown(f"""
                    <div class='food-card'>
                        <div style='text-align: center; font-size: 2.5rem; margin: 5px 0;'>{pred['emoji']}</div>
                        <h5 style='text-align: center; color: #2d3748; margin: 5px 0;'>{pred['name']}</h5>
                        <div class='nutrition-badge {box_cls}' style='width: 90%; margin: 5px auto; text-align: center; font-size: 0.9rem;'>
                            {pred['calories']} cal
                        </div>
                        <p style='text-align: center; font-size: 0.8rem; color: #718096; margin: 5px 0;'>
                            P: {pred['protein']}g | C: {pred['carbs']}g<br>F: {pred['fat']}g
                        </p>
                        <p style='text-align: center; font-size: 0.75rem; color: #a0aec0; margin-top: 5px;'>
                            ✅ {pred['confidence']:.0%} confidence
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

# ABOUT PAGE
elif st.session_state.page == 'about':
    st.markdown("# ℹ️ About This App")
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🍎 What is AI Food Nutrition Classifier?
        
        This application uses state-of-the-art computer vision and deep learning 
        to identify food items from images and provide detailed nutritional information.
        
        **Key Features:**
        - 🎯 **High Accuracy**: Powered by YOLOv11 model trained on diverse food datasets
        - 📊 **Detailed Nutrition**: Calorie count, protein, carbs, fat, fiber, and more
        - ⚡ **Real-time Analysis**: Get results in seconds
        - 📱 **Easy to Use**: Simple upload or camera capture
        - 📈 **Portion Control**: Adjust serving sizes for accurate calculations
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 How It Works
        
        1. **Upload/Capture**: Take a photo of your food or upload from gallery
        2. **AI Processing**: Our model analyzes the image using deep learning
        3. **Food Identification**: Recognizes the food item from 20+ categories
        4. **Nutrition Calculation**: Provides detailed nutritional information
        5. **Health Tips**: Get personalized recommendations
        
        ### 📚 Database
        
        Our nutrition database contains **static, scientifically accurate values** 
        for each food class, ensuring reliable calorie and nutrient information.
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🛠️ Technology Stack
        
        - **YOLOv11**: Object detection model
        - **PyTorch**: Deep learning framework
        - **Streamlit**: Web application framework
        - **OpenCV**: Image processing
        - **Pandas**: Data handling
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Nutrition Science
        
        All nutritional values are based on:
        - USDA Food Database
        - Scientific research papers
        - Standard food composition tables
        - Portion size guidelines
        """)
    
    with col3:
        st.markdown("""
        ### 📱 Future Features
        
        - Meal planning assistant
        - Daily calorie tracking
        - Recipe suggestions
        - Barcode scanner integration
        - Multi-food detection
        """)
    
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Disclaimer
    
    This application provides estimates based on image recognition. 
    For medical or precise dietary needs, consult a registered dietitian 
    or healthcare professional. Actual nutritional values may vary based 
    on preparation methods and specific ingredients.
    """)

# FOOTER
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🍎 <strong>AI Food Nutrition Classifier</strong> | Powered by Deep Learning</p>
        <p style='font-size: 0.9rem;'>For educational and informational purposes only</p>
        <p style='font-size: 0.8rem; opacity: 0.7;'>© 2024 FoodAI | Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)