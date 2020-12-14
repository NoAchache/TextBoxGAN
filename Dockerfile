FROM tensorflow/tensorflow:2.2.0-gpu
WORKDIR /TextBoxGan
COPY . .
RUN pip install -U pip
RUN pip install -r requirements.txt