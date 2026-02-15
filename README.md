# Symulacja Dyfuzji Ciepła w Mieszkaniu 2D

## Opis projektu
Projekt przedstawia symulację numeryczną rozkładu temperatury w mieszkaniu 2D w czasie. Model opiera się na rozwiązaniu równania przewodnictwa cieplnego (równanie dyfuzji) przy użyciu metod różnic skończonych.

Główne cele projektu to:
- Symulacja zmian temperatury w pomieszczeniach z uwzględnieniem strat ciepła przez ściany i okna.
- Analiza wpływu grzejników na rozkład temperatury.
- Obliczanie całkowitego zużycia energii ($\Psi$) w różnych scenariuszach ogrzewania (np. ogrzewanie ciągłe vs. okresowe).


## Opis zawartości modułów

### pipeline/

Folder zawiera serię skryptów .py, czyli logikę biznesową projektu:

-flat.py: Definiuje klasę Flat, która odpowiada za inicjalizację siatki, ustawienie warunków brzegowych (okna, drzwi, grzejniki) oraz wykonywanie kroków czasowych symulacji.
-Diff_matrices.py: Zawiera funkcje tworzące macierze niezbędne do obliczania drugiej pochodnej po zmiennych przestrzennych.
-load_constants.py: Skrypt obsługujący poprawne wczytywanie stałych fizycznych ze ścieżek systemowych.

### notebooks/

Folder zawiera notatniki Jupyter (.ipynb), które służą do przeprowadzania eksperymentów i wizualizacji wyników. Znajdują się tu wykresy temperatury w czasie oraz analizy zużycia energii dla różnych ustawień grzania grzejników.

###data/constants.csv

Plik konfiguracyjny zawierający stałe takie jak: współczynniki przewodnictwa ($\lambda$), ciepło właściwe, gęstość powietrza. 

## Instalacja i uruchomienie

Aby uruchomić projekt na lokalnej maszynie, wykonaj poniższe kroki:

1.Sklonuj repozytorium: \
Bashgit clone [https://github.com/TwojLogin/Projekt_mod_det_JK.git](https://github.com/TwojLogin/Projekt_mod_det_JK.git)
cd Projekt_mod_det_JK

2. Zainstaluj wymagane zależności:\
Upewnij się, że masz zainstalowanego Pythona. Następnie zainstaluj biblioteki z pliku requirements.txt:\
pip install -r requirements.txt

3. Uruchom symulacje:\
Najlepszym sposobem na przeglądanie wyników jest uruchomienie serwera Jupyter Notebook. W przeglądarce wejdź do folderu notebooks/ i wybierz interesujący Cię eksperyment.
