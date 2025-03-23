

# Flavour Match 🍽️
Flavour Match is a food recommendation system that suggests Indian cuisine recipes based on user dietary preferences, flavor profiles, and cooking time constraints.

📌 Features
✅ Recommends dishes based on:
* Dietary preferences (Vegetarian, Vegan, Non-Vegetarian, etc.)
* Flavor profiles (Spicy, Sweet, Savory, etc.)
* Cooking time constraints
✅ Uses machine learning to analyze ingredients, reviews, and ratings✅ Supports one-click filtering for customized recipe searches

📂 Dataset Details
The project uses a dataset containing Indian recipes with the following features:
* Recipe Name
* Description
* Ingredients & Quantity
* Preparation & Cooking Time
* Diet Type (Vegetarian, Vegan, etc.)
* Cuisine Type (North Indian, South Indian, etc.)
* User Reviews & Ratings
🛠️ Data Preprocessing Steps
* Handling missing values
* Standardizing column names
* Encoding categorical variables using One-Hot Encoding
* Normalizing numerical data

🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/arpitasingh16/Flavour-Match-.git
cd flavour-match
<br>
2️⃣ Create a Virtual Environment (Optional)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
<br>
3️⃣ Install Dependencies
pip install -r requirements.txt
<br>
4️⃣ Run the Project
jupyter notebook
Open basic.ipynb to start preprocessing the dataset.
<br>

🖥️ Usage
1️⃣ Load the dataset in basic.ipynb.
<br>
2️⃣ Preprocess the data using the provided steps.
<br>
3️⃣ Train the recommendation model.
<br>
4️⃣ Input user preferences and get personalized recipe recommendations!

🛠️ Technologies Used
* Python 🐍
* Pandas, NumPy (Data Processing)
* Scikit-learn (Machine Learning)
* Jupyter Notebook (Development)

🤝 Contributing
Want to improve Flavour Match? Follow these steps:
1. Fork the repository
2. Create a new branch (feature-improvement)
3. Commit your changes
4. Push to your fork
5. Create a pull request

📜 License
This project is open-source and available under the MIT License.
