# This includes Spark and Hadoop
FROM jupyter/pyspark-notebook:latest

# Set working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script into the container
COPY script.py /app/script.py

# Set the command to run your script
CMD ["spark-submit", "script.py"]