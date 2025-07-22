# Dockerfile dla projektu segmentacji bainitu
FROM continuumio/miniconda3:latest

WORKDIR /app

# Kopiuj pliki środowiska
COPY environment.yaml .
COPY requirements.txt .

# Stwórz środowisko conda
RUN conda env create -f environment.yaml

# Aktywuj środowisko
SHELL ["conda", "run", "-n", "bainite-segmentation", "/bin/bash", "-c"]

# Kopiuj kod źródłowy
COPY src/ ./src/

# Ustaw punkt wejścia
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bainite-segmentation"]
CMD ["python", "-m", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
