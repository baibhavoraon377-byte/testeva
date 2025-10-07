# ===============================================================
# Advanced DeepSeek Text Classification App (Optimized + SMOTE)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import warnings

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
nltk.download('stopwords')

# ===============================================================
# TEXT CLEANING
# ===============================================================
class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"[^a-z\s]", '', text)
        text = re.sub(r"\s+", ' ', text).strip()
        tokens = [w for w in text.split() if w not in self.stop_words]
        return " ".join(tokens)

# ===============================================================
# FEATURE EXTRACTION
# ===============================================================
class AdvancedFeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=3, max_df=0.9)
        self.svd = TruncatedSVD(n_components=150, random_state=42)
        self.selector = SelectKBest(mutual_info_classif, k=100)
        self.scaler = StandardScaler(with_mean=False)

    def fit_transform(self, texts, labels):
        tfidf_matrix = self.tfidf.fit_transform(texts)
        reduced = self.svd.fit_transform(tfidf_matrix)
        selected = self.selector.fit_transform(reduced, labels)
        scaled = self.scaler.fit_transform(selected)
        return scaled

    def transform(self, texts):
        tfidf_matrix = self.tfidf.transform(texts)
        reduced = self.svd.transform(tfidf_matrix)
        selected = self.selector.transform(reduced)
        scaled = self.scaler.transform(selected)
        return scaled

# ===============================================================
# MODEL TRAINER WITH SMOTE
# ===============================================================
class AdvancedModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', C=0.8, solver='liblinear'),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', max_depth=8, min_samples_split=5, min_samples_leaf=3),
            "Support Vector Machine": SVC(random_state=42, probability=True, class_weight='balanced', C=0.5, kernel='rbf', gamma='scale'),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, random_state=42, learning_rate=0.1, max_depth=4, min_samples_split=5),
            "Naive Bayes": MultinomialNB(alpha=0.5)
        }

    def enhanced_cross_validation(self, X, y, model, cv_folds=5):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
        return cv_scores.mean(), cv_scores.std()

    def train_and_evaluate(self, X, y):
        results = {}

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Apply SMOTE
        st.info("Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        st.success(f"SMOTE applied. Class distribution: {np.bincount(y_train)}")

        for name, model in self.models.items():
            st.markdown(f"### 🔹 Training {name}")
            model.fit(X_train, y_train)

            cv_mean, cv_std = self.enhanced_cross_validation(X_train, y_train, model)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }

            st.write(f"**Accuracy:** {acc:.3f} | **F1:** {f1:.3f} | **CV Mean:** {cv_mean:.3f} ± {cv_std:.3f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{name} - Confusion Matrix")
            st.pyplot(fig)

        return results

# ===============================================================
# STREAMLIT APP UI
# ===============================================================
st.set_page_config(page_title="DeepSeek NLP Classifier", layout="wide")

st.title("🧠 Advanced Text Classification with SMOTE (Improved Accuracy)")
st.markdown("### Upload CSV with columns: `Statement`, `Target`")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    if "Statement" not in df.columns or "Target" not in df.columns:
        st.error("CSV must contain 'Statement' and 'Target' columns!")
    else:
        cleaner = TextCleaner()
        df["CleanText"] = df["Statement"].apply(cleaner.clean)

        st.success("Text cleaned successfully!")

        extractor = AdvancedFeatureExtractor()
        X = extractor.fit_transform(df["CleanText"], df["Target"])
        y = df["Target"]

        trainer = AdvancedModelTrainer()
        results = trainer.train_and_evaluate(X, y)

        res_df = pd.DataFrame(results).T.sort_values("accuracy", ascending=False)
        st.dataframe(res_df.style.highlight_max(axis=0, color="lightgreen"))

        st.download_button(
            label="Download Results CSV",
            data=res_df.to_csv(index=True).encode('utf-8'),
            file_name="model_results.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a dataset to begin training.")
