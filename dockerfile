FROM python:3.9

WORKDIR /backend
COPY ./requirements.txt /backend/requirements.txt
RUN pip install -r /backend/requirements.txt

COPY /src /src
COPY /data /data

WORKDIR /src
CMD ["uvicorn", "api:app"]