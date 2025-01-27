# Base image
FROM python:3.10-slim

# Install PostgreSQL client and other necessary libraries
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install scikit-learn

# Set the working directory
WORKDIR /app

# Copy the code
COPY . .

# Default command
CMD ["python", "main.py"]

