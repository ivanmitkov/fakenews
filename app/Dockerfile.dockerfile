# STEP 1: Install base image. Optimized for Python.
FROM python:3.7.10


EXPOSE 80


#Use working directory /app
WORKDIR /app

#Copy all the content of current directory to /app
ADD . /app

#Installing required packages
RUN pip install -r requirements.txt


#Set environment variable
#ENV NAME OpentoAll



ENTRYPOINT python app.py
