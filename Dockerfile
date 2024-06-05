# Use Python 3.9 as the base image
FROM python:3.9

# Install PyTorch, torchvision, and torchaudio (GPU version)
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# Install other Python packages
RUN pip install torchdrug
RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install transformers

COPY data/ ./data/
COPY lib/ ./lib/
COPY atpbind_main.py .