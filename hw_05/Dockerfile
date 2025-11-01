FROM agrigorev/zoomcamp-model:2025

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml .

# Install dependencies using uv
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy the FastAPI service script (using service_docker.py which uses pipeline_v2.bin)
COPY service_docker.py service.py

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
