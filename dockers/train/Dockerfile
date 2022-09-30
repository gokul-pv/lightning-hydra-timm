FROM python:3.9.14-slim-bullseye

WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

CMD ["python", "src/train.py" ]
