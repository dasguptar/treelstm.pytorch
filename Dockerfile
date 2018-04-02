FROM ubuntu:16.04

MAINTAINER Riddhiman Dasgupta <riddhiman.dasgupta@gmail.com>

RUN apt-get update
RUN apt-get install -y --no-install-recommends git curl wget ca-certificates bzip2 unzip openjdk-8-jdk-headless
RUN apt-get -y autoclean && apt-get -y autoremove

RUN curl -o /root/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x /root/miniconda.sh && \
    /root/miniconda.sh -b && \
    rm /root/miniconda.sh && \
    /root/miniconda3/bin/conda clean -ya

ENV PATH /root/miniconda3/bin:$PATH
WORKDIR /root/treelstm.pytorch
COPY requirements.txt .
RUN ["/bin/bash", "-c", "pip install -r requirements.txt"]

CMD ["/bin/bash"]
