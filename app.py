import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page Title
st.title("Fake Job Recruitment Detection")

# Load Dataset
df = pd.read_csv("fake_job_postings.csv")
df = df.fillna("")

# Features and Target
X = df["description"]
y = df["fraudulent"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

# User Input
job_text = st.text_area("Paste Job Description")

# Prediction Button
if st.button("Check Job"):

    job_vector = vectorizer.transform([job_text])

    prediction = model.predict(job_vector)

    if prediction[0] == 1:
        st.error("Fake Job Posting")
    else:
        st.success("Genuine Job Posting")

st.subheader("Career Guidance Chatbot")

question = st.text_input("Ask Career Question")

if question:

    q = question.lower()

    if "python" in q:
        st.write("Learn Python, Pandas, NumPy and Machine Learning.")

    elif "data science" in q:
        st.write("Focus on Statistics, Python and Machine Learning.")

    elif "cyber security" in q:
        st.write("Learn Networking, Linux and Ethical Hacking.")

    else:
        st.write("Please ask about Python, Data Science or Cyber Security.")        