# Use an official Python runtime as a parent image.
# The 'slim' version is smaller and good for production.
FROM python:3.11-slim

# Set environment variables for best practices
# 1. Prevents Python from writing .pyc files to disc
# 2. Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Create a non-root user for security
RUN addgroup --system app && adduser --system --group app

# Copy the requirements file and install dependencies
# This is done in a separate step to leverage Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
# The --chown flag sets the owner of the copied file to the 'app' user.
COPY --chown=app:app hf_to_s3_streaming.py .

# Switch to the non-root user
USER app

# The command to run when the container starts
CMD ["python", "hf_to_s3_streaming.py"]
