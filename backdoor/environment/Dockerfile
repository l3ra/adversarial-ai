FROM registry.codeocean.com/codeocean/miniconda3:4.5.11-python3.7-cuda9.2-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade --no-cache-dir --upgrade-strategy=only-if-needed \
      keras==2.2.4 \
      numpy==1.16.4 \
      sklearn==0.0 \
      tensorflow==1.13.1
