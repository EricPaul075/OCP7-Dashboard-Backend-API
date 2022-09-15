FROM python:3.9

WORKDIR /backend
COPY ./requirements.txt /backend/requirements.txt
RUN pip install -r /backend/requirements.txt

COPY /src /src
COPY /data /data

WORKDIR /src
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]