FROM --platform=linux/amd64 nvidia/cuda:11.7.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.9 python3-pip

WORKDIR /src

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install torch-scatter==2.1.1+pt113cu117 torch-cluster==1.6.1+pt113cu117 torch-sparse==0.6.17+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
RUN pip install torchdrug==0.2.0
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install transformers

COPY data/ ./data/
COPY lib/ ./lib/
COPY atpbind_main.py ./