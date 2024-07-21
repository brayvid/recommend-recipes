import pandas as pd
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import numpy as np

# Load preprocessed datasets
@st.cache_data
def load_preprocessed_data():
    with open('data/recipes.pkl', 'rb') as f:
        recipes = pickle.load(f)

    with open('data/interactions.pkl', 'rb') as f:
        interactions = pickle.load(f)

    return recipes, interactions

recipes, interactions = load_preprocessed_data()

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('data/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

algo = load_model()

# Compute ingredient matrix
def compute_ingredient_matrix(recipes):
    mlb = MultiLabelBinarizer()
    ingredient_matrix = mlb.fit_transform(recipes['ingredients'])
    return ingredient_matrix, mlb.classes_

ingredient_matrix, mlb_classes = compute_ingredient_matrix(recipes)

def get_recommendations(pantry, num_recommendations=10):
    # Binarize the pantry ingredients using the pre-trained MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=mlb_classes)
    mlb.fit(recipes['ingredients'])  # Ensure it's fitted with the same classes
    pantry_vector = mlb.transform([pantry])[0]

    # Calculate similarity scores between the pantry and recipe ingredients
    similarity_scores = ingredient_matrix.dot(pantry_vector)

    # Get top recipe indices based on similarity scores
    top_indices = similarity_scores.argsort()[::-1]

    # Filter recipes to only include those with all ingredients in the pantry
    filtered_indices = []
    for idx in top_indices:
        recipe_ingredients = set(recipes.iloc[idx]['ingredients'])
        if recipe_ingredients.issubset(set(pantry)):
            filtered_indices.append(idx)
            if len(filtered_indices) == num_recommendations:
                break

    if not filtered_indices:
        return []

    # Predict ratings for the filtered recipes using the collaborative filtering model
    top_recipe_ids = recipes.iloc[filtered_indices]['id'].values
    predictions = [algo.predict('user', recipe_id).est for recipe_id in top_recipe_ids]

    # Combine predictions with similarity scores
    combined_scores = [(recipe_id, score, prediction) for recipe_id, score, prediction in zip(top_recipe_ids, similarity_scores[filtered_indices], predictions)]

    # Sort by prediction scores and return top recommendations
    combined_scores.sort(key=lambda x: x[2], reverse=True)
    top_recommendations = combined_scores[:num_recommendations]

    # Return the top recommended recipe names
    recommended_recipes = recipes[recipes['id'].isin([rec[0] for rec in top_recommendations])]['name'].values
    return recommended_recipes

# Streamlit app layout
st.title("Pantry-Based Recipe Recommender")

st.sidebar.header("Input Your Pantry Items")
pantry = st.sidebar.text_area("Enter pantry items separated by commas", "sugar, eggs, flour, butter, vanilla extract, baking soda")
pantry_list = [item.strip() for item in pantry.split(',')]

if st.sidebar.button("Get Recommendations"):
    suggestions = get_recommendations(pantry_list)

    if len(suggestions) > 0:
        st.write("You can make the following recipes:")
        for recipe in suggestions:
            st.write(f"- {recipe}")
    else:
        st.write("No recipes can be made with the current pantry items.")