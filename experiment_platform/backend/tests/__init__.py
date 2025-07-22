"""
Pakiet testów dla experiment_platform_backend
"""
import sys
import os

# Dodanie ścieżki do modułów projektu dla przypadku gdy testy są uruchamiane bezpośrednio
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
