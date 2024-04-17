import streamlit as st

from news_classification.modelling.classifiers.boosting import (
    BoostingClassifier,
)

classifier = BoostingClassifier.load()


def inference_app():
    st.set_page_config()
    st.markdown(
        """
        <style>
            .stApp {
            background-color: black;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.header("News Article Classification", divider="rainbow")

    # Blank space
    st.markdown("#")

    headline = st.text_input(
        "Insert news headline",
        key="placeholder",
    )

    if headline:
        prediction = classifier.predict([headline])[0]
        st.markdown("#")

        st.markdown(
            "<h3 style='text-align: center;'>Predicted Class</h3>",
            unsafe_allow_html=True,
        )

        st.write(
            f"<h1 style='text-align: center; color: green;'>{prediction}</h1>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    inference_app()
