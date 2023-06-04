# syntax=docker/dockerfile:1

FROM python:3.10-bullseye

WORKDIR /falcon-server

COPY  . .

RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

EXPOSE 5000

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]