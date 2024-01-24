# Use Python 3.9 as the base image
FROM python:3.9


# Install PyTorch, torchvision, and torchaudio (CPU versions)
RUN pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

# Install other Python packages
RUN pip install torchdrug
RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install transformers

COPY models.py /models.py
COPY tasks.py /tasks.py
COPY lmg_512_4.pt /model.pt

COPY script.py /script.py
