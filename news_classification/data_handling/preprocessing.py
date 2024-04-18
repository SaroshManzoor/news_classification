import logging

import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords


def _lemmatize(text: str, lemmatizer: WordNetLemmatizer) -> str:
    """
    Lemmatize all words in a given string.
    Note: only word level tokenization is being performed.

    :param text:
        string to be lemmatized

    :param lemmatizer:
        Any object which implements the lemmatize method.

    :return:
        string with all word lemmas
    """
    return " ".join([lemmatizer.lemmatize(token) for token in text.split()])


def _remove_stop_words(text: str, stop_words: list[str]) -> str:
    """
    Remove the words from the string which are provided in the
    stop_words argument

    :param text:
        string to be processed

    :param stop_words:
        list of words to remove

    :return:
        string with removed words (if any)
    """
    return " ".join([word for word in text.split() if word not in stop_words])


def preprocess_headlines(
    headlines: pd.Series, remove_stop_words=True, lemmatize=True
) -> pd.Series:
    """
    Performs basic clean-up of headlines. i.e.:
        - removing non-alphabet characters (except spaces)
        - removing multiple blank spaces

    Additionally, this method can also be used to discard all stop words
    and to lemmatize all headlines by using the corresponding input
    arguments.

    :param headlines:
        series containing new headlines
    :param remove_stop_words:
        boolean to remove or accept stop words
    :param lemmatize:
        boolean to enable lemmatization

    :return:
        preprocessed headlines
    """
    headlines = headlines.str.lower()

    logging.info("Cleaning-up article headlines.")

    noise_patterns = [
        r"[^a-z\s]",  # all non - alphabet characters except blank spaces
        # r"\b.\b",  # single isolated characters
        r"\s+",  # multiple consecutive spaces
    ]

    for pattern in noise_patterns:
        headlines = headlines.str.replace(pattern, " ", regex=True)

    if remove_stop_words:
        logging.info("Removing stop words.")

        headlines = headlines.apply(
            _remove_stop_words, stop_words=stopwords.words("english")
        )

    if lemmatize:
        logging.info("Lemmatizing.")

        headlines = headlines.apply(
            _lemmatize, lemmatizer=get_wordnet_lemmatizer()
        )

    return headlines


def get_wordnet_lemmatizer() -> WordNetLemmatizer:
    lemmatizer = WordNetLemmatizer()

    # Verifying that corresponding corpus is also downloaded
    try:
        lemmatizer.lemmatize("test token")
    except LookupError:
        import nltk

        nltk.download("wordnet")

    return lemmatizer
