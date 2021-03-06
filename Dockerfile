FROM tensorflow/tensorflow:2.3.0-gpu
WORKDIR /TextBoxGan
COPY requirements.txt .

RUN apt-get install p7zip
RUN apt-get install wget

#tensorflow-gpu throws an error when infering aster so uninstall it
RUN pip uninstall -y tensorflow-gpu
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install opencv-python-headless
