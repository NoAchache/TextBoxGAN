FROM tensorflow/tensorflow:2.3.0-gpu
WORKDIR /TextBoxGAN

RUN apt-get install -y apt-utils p7zip wget

#tensorflow-gpu throws an error when infering aster so uninstall it
RUN pip uninstall -y tensorflow-gpu
RUN pip install -U pip
RUN pip install poetry
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./
RUN poetry install

RUN pip install opencv-python-headless
