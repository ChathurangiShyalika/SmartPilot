FROM jupyter/datascience-notebook
# FROM nvidia/cuda:12.0.1-base-ubuntu20.04
# FROM tensorflow/tensorflow:latest-gpu-jupyter


COPY requirements.txt ./
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt

ENV JUPYTER_ENABLE_LAB=yes

COPY --chown=${NB_UID}:${NB_GID} . /home/jovyan/work
WORKDIR /home/jovyan/work

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/home/jovyan/work", "--allow-root"]
