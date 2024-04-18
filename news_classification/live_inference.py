import streamlit as st

from news_classification.config.pipeline_config import PipelineConfig
from news_classification.modelling.classifiers import (
    BaseClassifier,
    classifier_map,
)


def inference_app(classifier: BaseClassifier) -> None:
    """
    Renders a basic UI to interact with a trained classifier

    :param classifier:
        trained classifier
    :return:
    """
    # Setting dark theme
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

    # Text Input section
    headline = st.text_input(
        "Insert news headline",
        key="placeholder",
    )

    if headline:
        prediction = classifier.predict([headline])[0]
        st.markdown("#")

        st.markdown(
            "<h3 style='text-align: center;'>Predicted Category</h3>",
            unsafe_allow_html=True,
        )

        st.write(
            f"<h1 style='text-align: center; color: green;'>{prediction}</h1>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    config = PipelineConfig.load()

    inference_app(
        classifier_map[config.classifier_name].load(),
    )
