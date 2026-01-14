# Use Python 3.9
FROM python:3.9

# Set working directory
WORKDIR /code

# Copy requirements
COPY requirements.txt .

# Install dependencies (Force CPU version here)
RUN pip install --no-cache-dir --upgrade -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Open port 7860
EXPOSE 7860

# Start command
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
