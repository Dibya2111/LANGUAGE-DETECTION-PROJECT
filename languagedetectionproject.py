import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv('language.csv')

# Check for missing values
data.isnull().sum()

# Extract features and labels
x = np.array(data['Text'])
y = np.array(data['language'])

# Convert text into numerical representation
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model and vectorizer
import pickle
pickle.dump(model, open("language_model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

# Streamlit UI
st.title("Language Detection App")
st.write("Enter a text below to detect its language.")

user_input = st.text_area("Enter text:")

if st.button("Detect Language"):
    if user_input.strip():
        # Load model and vectorizer
        model = pickle.load(open("language_model.pkl", "rb"))
        cv = pickle.load(open("vectorizer.pkl", "rb"))
        
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)
        st.success(f"Detected Language: {output[0]}")
    else:
        st.warning("Please enter some text to analyze.")

# Deployment Instructions:
# 1. Install requirements: `pip install streamlit scikit-learn pandas numpy`
# 2. Run locally: `streamlit run script.py`
# 3. Deploy on Streamlit Cloud or similar hosting service.
