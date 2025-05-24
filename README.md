# Bainite Segmentation

Instrukcja tworzenia środowiska do projektu **Bainite Segmentation**.

## Tworzenie środowiska Conda

> **Uwaga:** W katalogu znajdują się przykładowe zdjęcia służące jedynie do testów. Docelowo katalog powinien być pusty lub zawierać wyłącznie Twoje własne dane.

1. Upewnij się, że masz zainstalowaną [Anaconda](https://www.anaconda.com/products/distribution) lub [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Przejdź do katalogu projektu:

    ```bash
    cd Bainitu_segmenation
    ```

3. Utwórz środowisko na podstawie pliku `environment.yaml`:

    ```bash
    conda env create -f environment.yaml
    ```

4. Aktywuj środowisko:

    ```bash
    conda activate bainite-segmentation
    ```

## Uruchamianie narzędzia Labelme

Aby uruchomić narzędzie **Labelme**, wpisz w terminalu:

```bash
labelme --nodata -O metadata raw_images 
```

W razie problemów z instalacją sprawdź wersję Conda i poprawność pliku `environment.yaml`.