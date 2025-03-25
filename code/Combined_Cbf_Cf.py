import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="FlavorFinder",
    page_icon="🍛",
    layout="wide",
)

# ---- UPDATED CSS STYLING ----
st.markdown("""
    <style>
        /* Global Styling */
        body {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .stTitle {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #ff9800;
        }
        .stSidebar {
            background-color: #252525;
            padding: 20px;
        }
        .stButton>button {
            background-color: #ff9800;
            color: white;
            font-size: 18px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #ff5722;
        }
        .food-card {
            border-radius: 15px;
            background-color: #282828;
            padding: 15px;
            margin: 10px;
            box-shadow: 3px 3px 10px rgba(255, 152, 0, 0.5);
        }
        
        /* Sidebar Styling */
        .sidebar-title {
            font-size: 26px !important;
            font-weight: bold !important;
            color: #ffcc00 !important;
        }
        .sidebar-subtitle {
            font-size: 22px !important;
            font-weight: bold !important;
            color: #ff9800 !important;
        }
        .sidebar-text {
            font-size: 18px !important;
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)




# ---- LOAD DATA ----
df = pd.read_csv('/Users/apple/Desktop/Minor_2_Work/Flavour-Match-/Dataset/filled_food_recipes_final.csv')
df.fillna('', inplace=True)

# ---- CONTENT-BASED FILTERING ----
df['combined_features'] = df['description'] + ' ' + df['ingredients_name']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_food(food_name, top_n=5):
    if food_name not in df['name'].values:
        return "Food item not found in dataset."
    
    food_index = df[df['name'] == food_name].index[0]
    similarity_scores = list(enumerate(cosine_sim[food_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return df.iloc[recommended_indices][['name', 'cuisine', 'course', 'image_url']]

# ---- COLLABORATIVE FILTERING ----
if 'user_id' not in df.columns or 'rating' not in df.columns:
    df['user_id'] = np.random.randint(1, 101, df.shape[0])  
    df['rating'] = np.random.randint(1, 6, df.shape[0])  

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'name', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)
accuracy.rmse(predictions)

def recommend_food_collaborative(user_id, top_n=5):
    unique_foods = df['name'].unique()
    predictions = [(food, model.predict(user_id, food).est) for food in unique_foods]
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_foods = [food for food, _ in predictions[:top_n]]
    return df[df['name'].isin(recommended_foods)][['name', 'cuisine', 'course', 'image_url']]

# ---- STREAMLIT UI ----
st.markdown("<h1 class='stTitle'>🍛 FlavorFinder: AI-Powered Food Recommendations</h1>", unsafe_allow_html=True)

# ---- STREAMLIT SIDEBAR ----
with st.sidebar:
    st.markdown("<p class='sidebar-title'>🔍 Choose Recommendation Type</p>", unsafe_allow_html=True)
    mode = st.radio("", ["Content-Based Filtering", "Collaborative Filtering"])
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<p class='sidebar-subtitle'>🎨 UI Customization</p>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-text'>Play around with UI settings soon! 🚀</p>", unsafe_allow_html=True)

# ---- CONTENT-BASED UI ----
if mode == "Content-Based Filtering":
    st.markdown("## 🍽️ Find Similar Dishes")
    food_item = st.selectbox("🔍 Select a food item:", df['name'].unique())

    if st.button("Get Recommendations"):
        with st.spinner("🔄 Finding delicious matches..."):
            recommendations = recommend_food(food_item)
        
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.markdown("## 🍛 Recommended Dishes")
            cols = st.columns(2)
            for index, row in recommendations.iterrows():
                with cols[index % 2]:  # Alternate column layout
                    st.markdown(f"<div class='food-card'><h3>{row['name']}</h3>", unsafe_allow_html=True)
                    st.write(f"**Cuisine:** {row['cuisine']}")
                    st.write(f"**Course:** {row['course']}")
                    if pd.notna(row['image_url']) and row['image_url'].startswith("http"):
                        st.image(row['image_url'], width=250)
                    st.write("</div>", unsafe_allow_html=True)

# ---- COLLABORATIVE FILTERING UI ----
elif mode == "Collaborative Filtering":
    st.markdown("## 👤 Personalized Recommendations")
    user_id = st.number_input("🔢 Enter User ID:", min_value=1, step=1)

    if st.button("Get Personalized Recommendations"):
        with st.spinner("🔄 Generating recommendations..."):
            recommendations = recommend_food_collaborative(user_id)

        if recommendations.empty:
            st.error("No recommendations available for this user.")
        else:
            st.markdown("## 🍛 Your Recommended Dishes")
            cols = st.columns(2)
            for index, row in recommendations.iterrows():
                with cols[index % 2]:  
                    st.markdown(f"<div class='food-card'><h3>{row['name']}</h3>", unsafe_allow_html=True)
                    st.write(f"**Cuisine:** {row['cuisine']}")
                    st.write(f"**Course:** {row['course']}")
                    if pd.notna(row['image_url']) and row['image_url'].startswith("http"):
                        st.image(row['image_url'], width=250)
                    st.write("</div>", unsafe_allow_html=True)
