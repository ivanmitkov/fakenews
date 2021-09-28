#install base image. optimized for python.
FROM python:3.7.10

#portal 80
EXPOSE 80

#use working directory /app
WORKDIR /app

#copy all the content of current directory to /app
ADD . /app

#installing required packages
RUN pip install -r requirements.txt

#entrypoint
ENTRYPOINT python app.py
