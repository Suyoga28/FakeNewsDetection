import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("📰 Fake News Detection using ML")
st.write("Upload a dataset and test the model with your own news!")

# Upload the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully!")

        st.write("### Sample Data:")
        st.dataframe(df.head())

        labels = df.label
        x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

        # TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)
        tfidf_test = tfidf_vectorizer.transform(x_test)

        # Model Training
        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train, y_train)
        y_pred = pac.predict(tfidf_test)
        score = accuracy_score(y_test, y_pred)

        st.subheader("🔍 Model Accuracy:")
        st.write(f"**{round(score*100, 2)}%**")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
        st.subheader("📊 Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # Optional: Text Input for Custom Prediction
        st.subheader("✏️ Test Custom News Article")
        user_input = st.text_area("Enter news content below:")
        if st.button("Predict"):
            if user_input.strip() != "":
                user_tfidf = tfidf_vectorizer.transform([user_input])
                prediction = pac.predict(user_tfidf)[0]
                st.success(f"🧠 Prediction: **{prediction}**")
            else:
                st.warning("Please enter some text to predict.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("📂 Please upload a `news.csv` file to begin.")
