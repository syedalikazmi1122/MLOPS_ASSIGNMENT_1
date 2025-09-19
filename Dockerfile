# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install 
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app
COPY setup.cfg ./
COPY tests ./tests
COPY README.md ./

# Expose port
EXPOSE 5001

# Run the app
CMD ["python", "app/app.py"]
