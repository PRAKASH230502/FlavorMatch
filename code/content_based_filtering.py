import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('/Users/apple/Desktop/Minor_2_Work/Flavour-Match-/Dataset/filled_food_recipes_final.csv')

df.fillna('', inplace=True)  # Fill missing values

df['combined_features'] = df['description'] + ' ' + df['ingredients_name']  # Combine relevant features

# Convert text data into TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_food(food_name, top_n=5):
    if food_name not in df['name'].values:
        return "Food item not found in dataset."
    
    food_index = df[df['name'] == food_name].index[0]
    similarity_scores = list(enumerate(cosine_sim[food_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return df.iloc[recommended_indices][['name', 'cuisine', 'course', 'image_url']]

# Streamlit UI
st.title("🍛 FlavorFinder: Indian Cuisine Recommendation System")

food_item = st.selectbox("🔍 Select a food item:", df['name'].unique())

if st.button("Get Recommendations"):
    recommendations = recommend_food(food_item)
    
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.subheader("🍽️ Top Recommended Dishes")
        for index, row in recommendations.iterrows():
            st.markdown(f"### {row['name']}")
            st.write(f"**Cuisine:** {row['cuisine']}")
            st.write(f"**Course:** {row['course']}")
            if pd.notna(row['image_url']) and row['image_url'].startswith("http"):
                st.image(row['image_url'], width=300)
            st.write("---")
