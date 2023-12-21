# FROM python:3.7.16
FROM python:3.9-slim-buster
# Copy the application code to the container
COPY . /app
# Set the working directory in the container to /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
# Expose the specified port for incoming traffic
EXPOSE $PORT
## To run the image
# Set the command to run the image when the container starts
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app 
## app:app means the app name in app.py and gunicorn cmd is used to run the docker image and will assign 4 worker with pid
## u can check netstat -a -o in cmd to see 0.0.0.0:5000 if we give 5000 as port to run the app





# FROM python:3.9-slim-buster

# RUN apt update -y && apt install awscli -y
# WORKDIR /app

# COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install gunicorn


# CMD ["python3", "app.py"]

# Use an official Python image as the base image







# # FROM python:3.7.16
# FROM python:3.9-slim-buster
# # Create a non-root user to run the application
# RUN adduser --disabled-password --gecos "" myuser

# # Set the working directory in the container to /app
# WORKDIR /app

# # Copy the application code to the container
# COPY . /app

# # Create the virtual environment and install the packages
# RUN python3 -m venv env
# RUN /app/env/bin/python3 -m pip install --upgrade pip
# RUN env/bin/pip install --no-cache-dir -r requirements.txt
# RUN env/bin/pip install gunicorn
# RUN env/bin/pip install Flask


# # Set the command to run when the container starts
# CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

# # Run the command as the non-root user
# USER myuser

# # Expose the specified port for incoming traffic
# EXPOSE $PORT



