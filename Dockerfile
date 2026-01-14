# Use Python 3.9
FROM python:3.9

# Set working directory to the container's root
WORKDIR /code

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application files
COPY . .

# Open port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]