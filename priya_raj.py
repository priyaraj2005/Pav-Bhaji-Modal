
import os
import re
import json
import argparse
from urllib.parse import urlparse

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def get_filename(url):
    if not url:
        return None
    return os.path.basename(urlparse(url).path)


def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"(?i)(?<!\w)#?pav[\s_-]?bhaji(?!\w)",
        " ",
        text
    )

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_model():

    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
                strip_accents="unicode"
            )
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=50000,
                sublinear_tf=True
            )
        )
    ])

    return Pipeline([
        ("features", features),
        (
            "clf",
            LinearSVC(
                C=0.1,
                class_weight="balanced",
                random_state=42
            )
        )
    ])


print("DriveBuddyAI Pav-Bhaji text classification model")
print("Final model configuration: TF-IDF + LinearSVC")
