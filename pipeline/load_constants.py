import csv

def load_constants(file_path):
    """
    Funkcja do ładowania potrzebnych stałych z pliku constants
    """
    constants = {}
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['constant_name']
            value = row['value']
            constants[name] = float(value)

    return constants