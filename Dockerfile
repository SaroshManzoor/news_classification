FROM python:3.9.1
ENV TZ="Europe/Berlin"

RUN apt-get update

RUN pip install poetry -U

COPY . /src

WORKDIR /src

RUN poetry config virtualenvs.in-project true
RUN poetry install --without dev
RUN poetry run python -m nltk.downloader wordnet

ENTRYPOINT ["./docker-entrypoint.sh"]

