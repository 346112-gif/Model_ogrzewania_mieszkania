import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

""" Wypisujemy potrzebne stałe """
lambda_door = 0.70          #Współczynnik przepuszczalności ciepła przez drzwi
lambda_wall = 0.25          #Współczynnik przepuszczalności ciepła przez ściany
lambda_window = 0.60        #Współczynnik przepuszczalności ciepła przez okna
air_heat_coeff = 0.025      #Współczynnik przepuszczalności ciepła przez powietrze
air_movement_coeff = 320    #Współczynnik symulujący ruch powietrza
lambda_air = air_movement_coeff * air_heat_coeff

K = 273.15  #Stała do konwersji stopni między Kelwinem, a Celsjuszem
r = 287.05  #Indywidualna stała gazowa dla suchego powietrza (J/(kg*K))
c = 1005    #Ciepło właściwe powietrza przy stałym ciśnieniu (J/(kg*K))
p = 101325  #ciśnienie atmosferyczne (Pa)
P = 6885    #przyjęta moc grzejników (W)

hx = 0.1 #Krok przestrzenny w metrach
ht = 10  #Krok czasowy w sekundach

"""Funkcje liczące macierze pochodnych"""
def D1_forward(n):
  D1 = np.eye(n, k=1) - np.eye(n)
  return D1
def D1_backward(n):
  D1 = -np.eye(n, k=-1) + np.eye(n)
  return D1
def D2(n):
  D2_ = -2 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
  return D2_


class Flat:
    """
    Klasa do symulacji procesu dyfuzji w mieszkaniu przechowująca informacje
     o rozmieszczeniu ścian, okien, drzwi i grzejników mieszkania. Liczy przede wszystkim
     średnią temperaturę, zużytą energię do ogrzania mieszkania i rozłożenie temperatury w mieszkaniu
     """
    def __init__(self, flat_width, flat_length, flat_height,
                 goal_temp, init_temp,
                 north_out_temp, east_out_temp,
                 south_out_temp, west_out_temp):

        self.radiators_heat_level = 0     #Poziom grzania grzejników (od 0 do 5)
        self.average_temp = init_temp + K #Średnia temperatura w Kelwinach

        self.goal_temp = goal_temp + K           #Średnia temperatura docelowa mieszkania
        self.north_out_temp = north_out_temp + K #Temperatura na zewnątrz od strony ściany północnej
        self.east_out_temp = east_out_temp + K   #Temperatura na zewnątrz od strony ściany wschodniej
        self.south_out_temp = south_out_temp + K #Temperatura na zewnątrz od strony ściany południowej
        self.west_out_temp = west_out_temp + K   #Temperatura na zewnątrz od strony ściany zachodniej

        #Tworzymy siatkę przedstawiającą temperaturę w mieszkaniu
        self.H = flat_height
        self.x, self.y = np.arange(0, flat_width, hx), np.arange(0, flat_length, hx)
        self.Nx, self.Ny = len(self.x), len(self.y)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.heat_sources = np.zeros(self.Nx * self.Ny)

        self.current_temp = np.ones(self.X.shape)*(init_temp + K) #Początkowo zakładamy, że temperatura
                                                                  #jest taka sama w całym mieszkaniu

        #Znajdujemy indeksy brzegowe
        self.idx_north_wall = np.where(self.Y == self.y[0], True, False).flatten()
        self.idx_east_wall = np.where(self.X == self.x[-1], True, False).flatten()
        self.idx_south_wall = np.where(self.Y == self.y[-1], True, False).flatten()
        self.idx_west_wall = np.where(self.X == self.x[0], True, False).flatten()

        #Tworzymy macierze identycznościowe
        self.id_Nx = np.eye(self.Nx)
        self.id_Ny = np.eye(self.Ny)
        self.id_Nxy = np.eye(self.Nx * self.Ny)

        #Tworzymy laplasjan
        self.D2x, self.D2y = D2(self.Nx), D2(self.Ny)
        self.laplacian = np.kron(self.id_Ny, self.D2x) / hx ** 2 + np.kron(self.D2y, self.id_Nx) / hx ** 2

        #Ustawiamy macierze potrzebne na brzegi
        self.Cy_north = -air_heat_coeff/lambda_wall * np.kron(D1_forward(self.Ny), self.id_Nx)/hx + self.id_Nxy
        self.Cx_east = air_heat_coeff/lambda_wall * np.kron(self.id_Ny, D1_backward(self.Nx))/hx + self.id_Nxy
        self.Cy_south = air_heat_coeff/lambda_wall * np.kron(D1_backward(self.Ny), self.id_Nx)/hx + self.id_Nxy
        self.Cx_west = -air_heat_coeff/lambda_wall * np.kron(self.id_Ny, D1_forward(self.Nx))/hx + self.id_Nxy

        #Środek macierzy dyfuzji
        self.rho = p / r
        self.alfa = lambda_air / self.rho / c
        self.A = self.id_Nxy - self.alfa * ht * self.laplacian

        #Pełna macierz dyfuzji dla pustego mieszkania
        self.A[self.idx_north_wall, :] = self.Cy_north[self.idx_north_wall, :]
        self.A[self.idx_east_wall, :] = self.Cx_east[self.idx_east_wall, :]
        self.A[self.idx_south_wall, :] = self.Cy_south[self.idx_south_wall, :]
        self.A[self.idx_west_wall, :] = self.Cx_west[self.idx_west_wall, :]

    def change_radiators_heat_level(self, new_heat_level):
        """Zmienia ustawienie pokrętła na grzejnikach"""
        self.radiators_heat_level = new_heat_level

    def add_inner_object(self, placement, dist_from_wall, thickness, start, end, lambda_type):
        """
        Funkcja do wstawiania ścian i drzwi wewnętrznych. Podajemy ułożenie obiektu (poziome lub pionowe),
        odległość od ściany zewnętrznej dla zerowych indeksów (ściana północna dla poziomego obiektu i ściana
        zachodnia dla pionowego), grubość obiektu, początek i koniec ściany (względem odpowiednio zachodniej
        czy północnej ściany) i współczynnik przenikalności ciepła obiektu.
        """

        ratio = lambda_type / lambda_air

        if placement == "vertical":
            M = (self.X >= dist_from_wall - thickness / 2) & (self.X <= dist_from_wall + thickness / 2) & \
                (self.Y >= start) & (self.Y <= end)
        elif placement == "horizontal":
            M = (self.Y >= dist_from_wall - thickness / 2) & (self.Y <= dist_from_wall + thickness / 2) & \
                (self.X >= start) & (self.X <= end)

        idx = M.flatten()

        self.A[idx, :] = self.id_Nxy[idx, :] - (self.alfa * ratio * ht * self.laplacian)[idx, :]


    def add_radiator(self, placement, thickness, start, end, dist_from_wall):
        """
        Funkcja dodająca grzejnik do siatki. Te same parametry co w 'add_inner_object'.
        Używa też stałych z zewnątrz potrzebnych do wyznaczenia ilości ciepła."""
        area = thickness * (end - start)
        heat_gain = ht * P * r/ (p * c * area)

        if placement == "vertical":
            M = (self.X == self.x[int(dist_from_wall / hx)]) & (self.Y >= start) & (self.Y <= end)
        elif placement == "horizontal":
            M = (self.Y == self.y[int(dist_from_wall / hx)]) & (self.X >= start) & (self.X <= end)

        self.heat_sources[M.flatten()] = heat_gain



    def add_outer_object(self, outer_wall, start, end, lambda_type):
        """
        Dodaje okno lub drzwi na wybraną ścianę zewnętrzną. Działanie podobne do 'add_inner_object'.
        """
        if outer_wall == "north":
            M = np.where((self.Y == self.y[0]) & (self.X >= start) & (self.X <= end), True, False)
            idx = M.flatten()

            self.A[idx, :] = (-air_heat_coeff/lambda_type *
                              np.kron(D1_forward(self.Ny), self.id_Nx)/hx + self.id_Nxy)[idx, :]

        elif outer_wall == "east":
            M = np.where((self.X == self.x[0]) & (self.Y >= start) & (self.Y <= end), True, False)
            idx = M.flatten()

            self.A[idx, :] = (air_heat_coeff/lambda_type *
                              np.kron(self.id_Ny, D1_backward(self.Nx))/hx + self.id_Nxy)[idx, :]

        elif outer_wall == "south":
            M = np.where((self.Y == self.y[-1]) & (self.X >= start) & (self.X <= end), True, False)
            idx = M.flatten()

            self.A[idx, :] = (air_heat_coeff/lambda_type *
                              np.kron(D1_backward(self.Ny), self.id_Nx)/hx + self.id_Nxy)[idx, :]

        elif outer_wall == "west":
            M = np.where((self.X == self.x[-1]) & (self.Y >= start) & (self.Y <= end), True, False)
            idx = M.flatten()

            self.A[idx, :] = (-air_heat_coeff/lambda_type *
                              np.kron(self.id_Ny, D1_forward(self.Nx))/hx + self.id_Nxy)[idx, :]


    def heat_up(self, T_total):
        """
        Funkcja symulująca dyfuzję w podanym czasie w godzinach (później konwertowanych na sekundy).
        """
        A = sp.csr_matrix(self.A)            #Macierz bez uwzględniania zer dla szybszych obliczeń
        u_current = self.current_temp.copy() #Operujemy na kopii siatki
        u_current = u_current.flatten()      #Spłaszczamy do wektora dla obliczeń
        T = int(T_total*3600)                #konwersja jednostek na sekundy

        total_kwh = 0                      #Całkowite zużycie energii przez grzejniki
        cell_area = hx * hx                #Powierzchnia jednej komórki na siatce
        cell_mass = self.rho * cell_area   #Masa jednej komórki na siatce

        heating = 1  #Parametr do symulacji działania termostatu

        for _ in range(int(T/ht)):
            #Stałe temperatry na zewnątrz mieszkania
            u_current[self.idx_north_wall] = self.north_out_temp
            u_current[self.idx_east_wall] = self.east_out_temp
            u_current[self.idx_south_wall] = self.south_out_temp
            u_current[self.idx_west_wall] = self.west_out_temp

            can_heat = (u_current < 70 + K)  #Temperatura wody w grzejnikach
                                             # nie przekracza 70 stopni Celsjusza

            #Ciepło dodawane do komórek grzejników na siatce
            heat_added = self.heat_sources * (self.radiators_heat_level / 5) * can_heat * heating
            u_current += heat_added

            #obliczanie zużytej energii
            step_joules = np.sum(heat_added) * cell_mass * c
            total_kwh += step_joules / 3_600_000

            #dyfuzja w jednym kroku czasowym
            u_current = sp.linalg.spsolve(A, u_current)

            #Nowa średnia temperatura (bez indeksów brzegowych, które zawierają temperatury z zewnątrz
            self.average_temp = np.mean(u_current.reshape(self.Ny, self.Nx)[1:-1, 1:-1])

            #Symulowwane działanie termostatu
            if self.average_temp > self.goal_temp:
                heating = 0
            elif self.average_temp < self.goal_temp - 0.2:
                heating = 1

        self.current_temp = u_current.reshape(self.Ny, self.Nx) #Powrót do kształtu siatki

        #Wypisywanie danych
        average_temp = round(self.average_temp - K, 2)
        print(f"Czas trwania symulacji dyfuzji: {T_total}")
        print(f"Całkowita zużyta energia w trakcie symulacji dyfuzji: {total_kwh}")
        print(f"Średnia temperatura po symulacji dyfuzji: {average_temp}")

        return T_total, total_kwh, average_temp




    def temp_plot(self):
        """
        Funkcja rysująca rozkład temperatury w pomieszczeniu w Celsjuszach.
        """
        u_celc = self.current_temp - K

        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        z_min = 10
        z_max = 30
        levels = np.linspace(z_min, z_max, int(2*(z_max-z_min)))

        im1 = axs.contourf(self.X, self.Y, u_celc, levels=levels, cmap='viridis')
        fig.colorbar(im1, ax=axs, ticks=np.linspace(z_min, z_max, 6))

        plt.tight_layout()
        plt.gca().invert_yaxis() #odwracamy pionową oś, bo dla macierzy punkt
                                 #(0, 0) jest u góry po lewej
        plt.show()

