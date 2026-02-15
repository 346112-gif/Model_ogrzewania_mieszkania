import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

try:
    from pipeline.load_constants import load_constants
except ModuleNotFoundError:
    from load_constants import load_constants

try:
    from pipeline.Diff_matrices import D2, D1_forward, D1_backward
except ModuleNotFoundError:
    from Diff_matrices import D2, D1_forward, D1_backward
import os


class Flat:
    """
    Klasa do symulacji procesu dyfuzji w mieszkaniu przechowująca informacje
    o rozmieszczeniu ścian, okien, drzwi i grzejników mieszkania. Liczy przede wszystkim
    średnią temperaturę, zużytą energię do ogrzania mieszkania i rozłożenie temperatury w mieszkaniu
    """
    def __init__(self, flat_width, flat_length,
                 goal_temp, init_temp,
                 north_out_temp, east_out_temp,
                 south_out_temp, west_out_temp, time_step):

        # Pobiera ścieżkę do folderu 'pipeline' (tam gdzie jest flat.py)
        base_path = os.path.dirname(__file__)
        # Wychodzi poziom wyżej i wchodzi do 'data/constants.csv'
        path = os.path.abspath(os.path.join(base_path, "..", "data", "constants.csv"))
        data = load_constants(path)

        # Potrzebne stałe
        self.lambda_door = data["lambda_door"]               # Współczynnik przepuszczalności ciepła przez drzwi
        self.lambda_wall = data["lambda_wall"]               # Współczynnik przepuszczalności ciepła przez ściany
        self.lambda_window = data["lambda_window"]           # Współczynnik przepuszczalności ciepła przez okna
        self.air_heat_coeff = data["air_heat_coeff"]         # Współczynnik przepuszczalności ciepła przez powietrze
        self.air_movement_coeff = data["air_movement_coeff"] # Współczynnik symulujący ruch powietrza

        self.K = data["K"] # Stała do konwersji stopni między Kelwinem, a Celsjuszem
        self.r = data["r"] # Indywidualna stała gazowa dla suchego powietrza (J/(kg*K))
        self.c = data["c"] # Ciepło właściwe powietrza przy stałym ciśnieniu (J/(kg*K))
        self.p = data["p"] # Ciśnienie atmosferyczne (Pa)

        self.hx = 0.1        # Krok przestrzenny w metrach
        self.ht = time_step  # Krok czasowy w sekundach

        self.lambda_air = self.air_movement_coeff * self.air_heat_coeff

        self.radiators_heat_level = 0        # Poziom grzania grzejników (od 0 do 5)
        self.init_temp = init_temp  + self.K # Zamiana na stopnie Kelwina
        self.average_temp = self.init_temp   # Średnia temperatura w Kelwinach

        self.goal_temp = goal_temp + self.K           # Średnia temperatura docelowa mieszkania
        self.north_out_temp = north_out_temp + self.K # Temperatura na zewnątrz od strony ściany północnej
        self.east_out_temp = east_out_temp + self.K   # Temperatura na zewnątrz od strony ściany wschodniej
        self.south_out_temp = south_out_temp + self.K # Temperatura na zewnątrz od strony ściany południowej
        self.west_out_temp = west_out_temp + self.K   # Temperatura na zewnątrz od strony ściany zachodniej

        # Tworzymy siatkę przedstawiającą temperaturę w mieszkaniu
        self.x, self.y = np.arange(0, flat_width, self.hx), np.arange(0, flat_length, self.hx)
        self.Nx, self.Ny = len(self.x), len(self.y)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.heat_sources = np.zeros(self.Nx * self.Ny)
        self.current_temp = np.ones(self.X.shape) * self.init_temp  # Początkowo zakładamy, że temperatura
                                                                    # jest taka sama w całym mieszkaniu

        # Znajdujemy indeksy brzegowe
        self.idx_north_wall = np.where(self.Y == self.y[0], True, False).flatten()
        self.idx_east_wall = np.where(self.X == self.x[-1], True, False).flatten()
        self.idx_south_wall = np.where(self.Y == self.y[-1], True, False).flatten()
        self.idx_west_wall = np.where(self.X == self.x[0], True, False).flatten()

        # Tworzymy macierze identycznościowe
        self.id_Nx = np.eye(self.Nx)
        self.id_Ny = np.eye(self.Ny)
        self.id_Nxy = np.eye(self.Nx * self.Ny)

        # Tworzymy laplasjan
        self.D2x, self.D2y = D2(self.Nx), D2(self.Ny)
        self.laplacian = (np.kron(self.id_Ny, self.D2x) /
                          self.hx ** 2 + np.kron(self.D2y, self.id_Nx) / self.hx ** 2)

        # Ustawiamy macierze potrzebne na brzegi
        self.Cy_north = (-self.air_heat_coeff/self.lambda_wall *
                         np.kron(D1_forward(self.Ny), self.id_Nx)/self.hx + self.id_Nxy)
        self.Cx_east = (self.air_heat_coeff/self.lambda_wall *
                        np.kron(self.id_Ny, D1_backward(self.Nx))/self.hx + self.id_Nxy)
        self.Cy_south = (self.air_heat_coeff/self.lambda_wall *
                         np.kron(D1_backward(self.Ny), self.id_Nx)/self.hx + self.id_Nxy)
        self.Cx_west = (-self.air_heat_coeff/self.lambda_wall *
                        np.kron(self.id_Ny, D1_forward(self.Nx))/self.hx + self.id_Nxy)

        # Środek macierzy dyfuzji
        self.rho = self.p / self.r
        self.alfa = self.lambda_air / self.rho / self.c
        self.A = self.id_Nxy - self.alfa * self.ht * self.laplacian

        # Pełna macierz dyfuzji dla pustego mieszkania
        self.A[self.idx_north_wall, :] = self.Cy_north[self.idx_north_wall, :]
        self.A[self.idx_east_wall, :] = self.Cx_east[self.idx_east_wall, :]
        self.A[self.idx_south_wall, :] = self.Cy_south[self.idx_south_wall, :]
        self.A[self.idx_west_wall, :] = self.Cx_west[self.idx_west_wall, :]

    def change_radiators_heat_level(self, new_heat_level):
        """Zmienia ustawienie pokrętła na grzejnikach"""
        self.radiators_heat_level = new_heat_level

    def add_inner_object(self, placement, dist_from_wall, thickness, start, end, object_type):
        """
        Funkcja do wstawiania ścian i drzwi wewnętrznych. Podajemy ułożenie obiektu (poziome lub pionowe),
        odległość od ściany zewnętrznej dla zerowych indeksów (ściana północna dla poziomego obiektu i ściana
        zachodnia dla pionowego), grubość obiektu, początek i koniec ściany (względem odpowiednio zachodniej
        czy północnej ściany) i współczynnik przenikalności ciepła obiektu.
        """
        if object_type == "wall":
            lambda_type = self.lambda_wall
        elif object_type == "door":
            lambda_type = self.lambda_door
        else:
            raise ValueError("object_type has to be either 'wall' or 'door'")

        ratio = lambda_type / self.lambda_air

        if placement == "vertical":
            M = (self.X >= dist_from_wall - thickness / 2) & (self.X <= dist_from_wall + thickness / 2) & \
                (self.Y >= start) & (self.Y <= end)
        elif placement == "horizontal":
            M = (self.Y >= dist_from_wall - thickness / 2) & (self.Y <= dist_from_wall + thickness / 2) & \
                (self.X >= start) & (self.X <= end)
        else:
            raise ValueError("Placement has to be either 'vertical' or 'horizontal'")

        idx = M.flatten()

        self.A[idx, :] = self.id_Nxy[idx, :] - (self.alfa * ratio * self.ht * self.laplacian)[idx, :]


    def add_radiator(self, placement, thickness, start, end, dist_from_wall, power):
        """
        Funkcja dodająca grzejnik do siatki. Te same parametry co w 'add_inner_object'.
        Używa też podanej mocy i stałych z zewnątrz potrzebnych do wyznaczenia ilości ciepła.
        """
        area = thickness * (end - start)
        heat_gain = self.ht * power * self.r/ (self.p * self.c * area)

        if placement == "vertical":
            M = (self.X == self.x[int(dist_from_wall / self.hx)]) & (self.Y >= start) & (self.Y <= end)
        elif placement == "horizontal":
            M = (self.Y == self.y[int(dist_from_wall / self.hx)]) & (self.X >= start) & (self.X <= end)
        else:
            raise ValueError("Placement has to be either 'vertical' or 'horizontal'")

        self.heat_sources[M.flatten()] = heat_gain



    def add_outer_object(self, outer_wall, start, end, object_type):
        """
        Dodaje okno lub drzwi na wybraną ścianę zewnętrzną. Działanie podobne do 'add_inner_object'.
        """
        if object_type == "window":
            lambda_type = self.lambda_window
        elif object_type == "door":
            lambda_type = self.lambda_door
        else:
            raise ValueError("object_type has to be either 'window' or 'door'")

        if outer_wall == "north":
            M = np.where((self.Y == self.y[0]) & (self.X >= start) & (self.X <= end), True, False)
            idx = M.flatten()

            self.A[idx, :] = (-self.air_heat_coeff/lambda_type *
                              np.kron(D1_forward(self.Ny), self.id_Nx)/self.hx + self.id_Nxy)[idx, :]

        elif outer_wall == "east":
            M = np.where((self.X == self.x[0]) & (self.Y >= start) & (self.Y <= end), True, False)
            idx = M.flatten()

            self.A[idx, :] = (self.air_heat_coeff/lambda_type *
                              np.kron(self.id_Ny, D1_backward(self.Nx))/self.hx + self.id_Nxy)[idx, :]

        elif outer_wall == "south":
            M = np.where((self.Y == self.y[-1]) & (self.X >= start) & (self.X <= end), True, False)
            idx = M.flatten()

            self.A[idx, :] = (self.air_heat_coeff/lambda_type *
                              np.kron(D1_backward(self.Ny), self.id_Nx)/self.hx + self.id_Nxy)[idx, :]

        elif outer_wall == "west":
            M = np.where((self.X == self.x[-1]) & (self.Y >= start) & (self.Y <= end), True, False)
            idx = M.flatten()

            self.A[idx, :] = (-self.air_heat_coeff/lambda_type *
                              np.kron(self.id_Ny, D1_forward(self.Nx))/self.hx + self.id_Nxy)[idx, :]


    def heat_up(self, T_total):
        """
        Funkcja symulująca dyfuzję w podanym czasie w godzinach (później konwertowanych na sekundy).
        """
        A = sp.csr_matrix(self.A)            # Macierz bez uwzględniania zer dla szybszych obliczeń
        u_current = self.current_temp.copy() # Operujemy na kopii siatki
        u_current = u_current.flatten()      # Spłaszczamy do wektora dla obliczeń
        T = int(T_total*3600)                # konwersja jednostek na sekundy

        total_kwh = 0                      # Całkowite zużycie energii przez grzejniki
        cell_area = self.hx ** 2           # Powierzchnia jednej komórki na siatce
        cell_mass = self.rho * cell_area   # Masa jednej komórki na siatce

        heating = 0  # Parametr do symulacji działania termostatu

        for _ in range(int(T/self.ht)):
            # Stałe temperatry na zewnątrz mieszkania
            u_current[self.idx_north_wall] = self.north_out_temp
            u_current[self.idx_east_wall] = self.east_out_temp
            u_current[self.idx_south_wall] = self.south_out_temp
            u_current[self.idx_west_wall] = self.west_out_temp

            can_heat = (u_current < 70 + self.K)  # Temperatura wody w grzejnikach
                                                  # nie przekracza 70 stopni Celsjusza

            # Ciepło dodawane do komórek grzejników na siatce
            heat_added = self.heat_sources * (self.radiators_heat_level / 5) * can_heat * heating
            u_current += heat_added

            # Obliczanie zużytej energii
            step_joules = np.sum(heat_added) * cell_mass * self.c
            total_kwh += step_joules / 3_600_000

            # Dyfuzja w jednym kroku czasowym
            u_current = sp.linalg.spsolve(A, u_current)

            # Nowa średnia temperatura (bez indeksów brzegowych, które zawierają temperatury z zewnątrz
            self.average_temp = np.mean(u_current.reshape(self.Ny, self.Nx)[1:-1, 1:-1])

            # Symulowwane działanie termostatu
            if self.average_temp > self.goal_temp:
                heating = 0
            elif self.average_temp < self.goal_temp - 0.2:
                heating = 1

        self.current_temp = u_current.reshape(self.Ny, self.Nx) #Powrót do kształtu siatki

        # Wypisywanie danych
        average_temp = round(self.average_temp - self.K, 2)

        return total_kwh, average_temp




    def temp_plot(self):
        """
        Funkcja rysująca rozkład temperatury w pomieszczeniu w Celsjuszach.
        """
        u_celc = self.current_temp - self.K

        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        z_min = 10
        z_max = 30
        levels = np.linspace(z_min, z_max, int(2*(z_max-z_min)))

        im1 = axs.contourf(self.X, self.Y, u_celc, levels=levels, cmap='viridis')
        fig.colorbar(im1, ax=axs, ticks=np.linspace(z_min, z_max, 6))

        plt.tight_layout()
        plt.gca().invert_yaxis() # odwracamy pionową oś, bo dla macierzy punkt
                                 # (0, 0) jest u góry po lewej
        plt.show()


