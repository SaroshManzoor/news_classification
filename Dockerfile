FROM python:3.9.1
ENV TZ="Europe/Berlin"

RUN apt-get update

RUN pip install poetry -U

## This is required for LGBM to work properly
ENV LD_PRELOAD='/usr/lib/aarch64-linux-gnu/libgomp.so.1'

COPY . /src

WORKDIR /src

RUN poetry config virtualenvs.in-project true
RUN poetry install
RUN poetry run python -m nltk.downloader wordnet
RUN poetry run pytest

ENTRYPOINT ["./docker-entrypoint.sh"]

