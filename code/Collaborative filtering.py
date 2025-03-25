import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load dataset
df = pd.read_csv('/Users/apple/Desktop/Minor_2_Work/Flavour-Match-/Dataset/filled_food_recipes_final.csv')

print(df.columns)

import numpy as np

# Generate synthetic user_id (assuming at least 100 users)
df['user_id'] = np.random.randint(1, 101, df.shape[0])

# Generate synthetic ratings between 1-5
df['rating'] = np.random.randint(1, 6, df.shape[0])

print(df.head())  # Check the first few rows
print(df.info())  # Check if 'user_id' and 'rating' exist

if 'user_id' not in df.columns or 'rating' not in df.columns:
    st.error("Dataset does not contain user_id or rating columns required for collaborative filtering.")
    st.stop()

# Define the Reader for Surprise
reader = Reader(rating_scale=(1, 5))

# Load data into Surprise format
data = Dataset.load_from_df(df[['user_id', 'name', 'rating']], reader)

# Split the dataset into training and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD Model (Singular Value Decomposition)
model = SVD()
model.fit(trainset)

# Evaluate the model
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# Function to get recommendations
def recommend_food_collaborative(user_id, top_n=5):
    """Recommend top N food items for a given user using collaborative filtering."""
    
    unique_foods = df['name'].unique()
    
    predictions = []
    for food in unique_foods:
        predictions.append((food, model.predict(user_id, food).est))
    
    # Sort recommendations by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    recommended_foods = [food for food, _ in predictions[:top_n]]
    return df[df['name'].isin(recommended_foods)][['name', 'cuisine', 'course', 'image_url']]

# Streamlit UI
st.title("🍛 FlavorFinder: Collaborative Filtering Recommendation System")

user_id = st.number_input("🔢 Enter User ID:", min_value=1, step=1)

if st.button("Get Personalized Recommendations"):
    recommendations = recommend_food_collaborative(user_id)
    
    if recommendations.empty:
        st.error("No recommendations available for this user.")
    else:
        st.subheader("🍽️ Personalized Food Recommendations")
        for index, row in recommendations.iterrows():
            st.markdown(f"### {row['name']}")
            st.write(f"**Cuisine:** {row['cuisine']}")
            st.write(f"**Course:** {row['course']}")
            if pd.notna(row['image_url']) and row['image_url'].startswith("http"):
                st.image(row['image_url'], width=300)
            st.write("---")
