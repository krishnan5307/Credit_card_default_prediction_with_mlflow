FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]





# Use an official Python image as the base image
FROM python:3.7.16

# Create a non-root user to run the application
RUN adduser --disabled-password --gecos "" myuser

# Set the working directory in the container to /app
WORKDIR /app

# Copy the application code to the container
COPY . .

# Create the virtual environment and install the packages
RUN python3 -m venv env
RUN /app/env/bin/python3 -m pip install --upgrade pip
RUN env/bin/pip install --no-cache-dir -r requirements.txt

# Set the command to run when the container starts
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

# Run the command as the non-root user
USER myuser

# Expose the specified port for incoming traffic
EXPOSE $PORT



