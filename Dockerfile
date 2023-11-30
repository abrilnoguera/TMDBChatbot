FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install gunicorn   # Add this line to install gunicorn

RUN mkdir /.surprise_data && chmod -R 777 /.surprise_data

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
