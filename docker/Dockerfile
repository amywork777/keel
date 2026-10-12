# Dev/test image: NVIDIA's usd-brep kernel only ships Linux/Windows wheels.
FROM python:3.12-slim
RUN pip install --no-cache-dir --root-user-action=ignore usd-brep pytest numpy
ENV PYTHONPATH=/keel/src PYTHONDONTWRITEBYTECODE=1
WORKDIR /keel
