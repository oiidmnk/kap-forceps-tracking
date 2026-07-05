FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tool_positions.py ws_server.py input_example.json ./

EXPOSE 8765

CMD ["python", "ws_server.py", "--host", "0.0.0.0", "--port", "8765"]
