FROM jdzw2014/anaconda:1.0

ARG CONDA=/root/anaconda3/bin/conda

ARG PROJECT_DIR=/2021-top-data-competition
COPY ./ $PROJECT_DIR
ENV PYTHONPATH="${PYTHONPATH}:$PROJECT_DIR/1-shopee同款商品选择最低价"

RUN $CONDA env create -f environment.yaml
