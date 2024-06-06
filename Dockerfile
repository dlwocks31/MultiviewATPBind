# Use Python 3.9 as the base image
FROM --platform=linux/amd64 python:3.9

# Install PyTorch, torchvision, and torchaudio (GPU version)
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install torch-scatter==2.1.1+pt113cu117 torch-cluster==1.6.1+pt113cu117 torch-sparse==0.6.17+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
# Install other Python packages
RUN pip install torchdrug==0.2.0
RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install transformers

COPY data/ ./data/
COPY lib/ ./lib/
COPY atpbind_main.py .