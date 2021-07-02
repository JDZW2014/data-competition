FROM jdzw2014/anaconda:1.0

ARG CONDA=/root/anaconda3/bin/conda
ARG PIP=/root/anaconda3/bin/pip

ARG PROJECT_DIR=/2021-top-data-competition
COPY ./ $PROJECT_DIR
ENV PYTHONPATH="${PYTHONPATH}:$PROJECT_DIR/1-shopee-goods-match-competition"

RUN $CONDA install --yes --file /2021-top-data-competition/1-shopee-goods-match-competition/requirements_conda.txt
RUN $PIP install -r /2021-top-data-competition/1-shopee-goods-match-competition/requirements_pip.txt

