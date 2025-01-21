# Use a lightweight Python image
FROM python:3.10-slim

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers torch litgpt lightning dataclasses pydantic 

# Copy application files
COPY app /app

# Set working directory
WORKDIR /app

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-max-requests", "1000000000"]