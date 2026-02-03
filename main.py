import numpy as np
import matplotlib.pyplot as plt
from time import time

""" Wypisujemy potrzebne stałe """
lambda_door = 0.80
lambda_wall = 0.30
lambda_window = 0.90
lambda_air = 3
r = 287.05 #Indywidualna stała gazowa dla suchego powietrza (J/(kg*K))
c = 1005 #Ciepło właściwe powietrza przy stałym ciśnieniu (J/(kg*K))
p = 101325 #ciśnienie atmosferyczne (Pa)
P = 1500 #przyjęta moc grzejników (W)

"""kroki przestrzenne i czasowe"""
hx = 0.2
hy = 0.2
ht = 5

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
    """Przy tworzeniu mieszkania podajemy jego szerokość i długość w metrach, a docelową temperaturę w stopniach celsjusza"""
    def __init__(self, flat_width, flat_length,
                 goal_temp, init_temp,
                 north_out_temp, east_out_temp,
                 south_out_temp, west_out_temp):
        self.radiators_heat_level = 0
        self.average_temp = init_temp + 273.15

        self.goal_temp = goal_temp
        self.north_out_temp = north_out_temp + 273.15
        self.east_out_temp = east_out_temp + 273.15
        self.south_out_temp = south_out_temp + 273.15
        self.west_out_temp = west_out_temp + 273.15


        self.x, self.y = np.arange(0, flat_width, hx), np.arange(0, flat_length, hy)
        self.Nx, self.Ny = len(self.x), len(self.y)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.heat_sources = np.zeros(self.Nx * self.Ny)

        self.current_temp = np.ones(self.X.shape)*(init_temp + 273.15) #Początkowo zakładamy, że temperatura
                                                                         #jest taka sama w całym mieszkaniu

        self.idx_north_wall = np.where(self.Y == self.y[0], True, False).flatten()
        self.idx_east_wall = np.where(self.X == self.x[-1], True, False).flatten()
        self.idx_south_wall = np.where(self.Y == self.y[-1], True, False).flatten()
        self.idx_west_wall = np.where(self.X == self.x[0], True, False).flatten()

        self.id_Nx = np.eye(self.Nx)
        self.id_Ny = np.eye(self.Ny)
        self.id_Nxy = np.eye(self.Nx * self.Ny)

        self.D2x, self.D2y = D2(self.Nx), D2(self.Ny)
        self.laplacian = np.kron(self.id_Ny, self.D2x) / hx ** 2 + np.kron(self.D2y, self.id_Nx) / hy ** 2

        self.Cy_north = -lambda_air/lambda_wall * np.kron(D1_forward(self.Ny), self.id_Nx)/hy + self.id_Nxy
        self.Cx_east = lambda_air/lambda_wall * np.kron(self.id_Ny, D1_backward(self.Nx))/hx + self.id_Nxy
        self.Cy_south = lambda_air/lambda_wall * np.kron(D1_backward(self.Ny), self.id_Nx)/hy + self.id_Nxy
        self.Cx_west = -lambda_air/lambda_wall * np.kron(self.id_Ny, D1_forward(self.Nx))/hx + self.id_Nxy

        self.rho = p / (r * self.average_temp)
        self.alfa = lambda_air / self.rho / c
        self.A = self.id_Nxy - self.alfa * ht * self.laplacian

        self.A[self.idx_north_wall, :] = self.Cy_north[self.idx_north_wall, :]
        self.A[self.idx_east_wall, :] = self.Cx_east[self.idx_east_wall, :]
        self.A[self.idx_south_wall, :] = self.Cy_south[self.idx_south_wall, :]
        self.A[self.idx_west_wall, :] = self.Cx_west[self.idx_west_wall, :]


    def change_radiators_heat_level(self, new_heat_level):
        self.radiators_heat_level = new_heat_level

    def add_inner_object(self, placement, dist_from_wall, start, end, lambda_type):
        ratio = lambda_type / lambda_air

        if placement == "vertical":
            M = (self.X == self.x[int(dist_from_wall / hx)]) & (self.Y >= start) & (self.Y <= end)
        elif placement == "horizontal":
            M = (self.Y == self.y[int(dist_from_wall / hy)]) & (self.X >= start) & (self.X <= end)

        idx = M.flatten()

        self.A[idx, :] = self.id_Nxy[idx, :] - (self.alfa * ratio * ht * self.laplacian)[idx, :]

    def add_radiator(self, placement, thickness, start, end, dist_from_wall):
        area = thickness * (end - start)
        heat_gain = ht * P * r / (p * c * area)

        if placement == "vertical":
            M = (self.X == self.x[int(dist_from_wall / hx)]) & (self.Y >= start) & (self.Y <= end)
        elif placement == "horizontal":
            M = (self.Y == self.y[int(dist_from_wall / hy)]) & (self.X >= start) & (self.X <= end)

        self.heat_sources[M.flatten()] = heat_gain



    def add_outer_object(self, outer_wall, start, end, lambda_type):
        if outer_wall == "north":
            M = (self.Y == self.y[0]) & (self.X >= int(start/hx)) & (self.X <= int(end/hx))
            idx = M.flatten()

            self.A[idx, :] = lambda_wall/lambda_type * self.Cy_north[idx, :]

        elif outer_wall == "east":
            M = (self.X == self.x[0]) & (self.Y >= int(start / hx)) & (self.Y <= int(end / hx))
            idx = M.flatten()

            self.A[idx, :] = lambda_wall/lambda_type * self.Cx_east[idx, :]

        elif outer_wall == "south":
            M = (self.Y == self.y[-1]) & (self.X >= int(start / hx)) & (self.X <= int(end / hx))
            idx = M.flatten()

            self.A[idx, :] = lambda_wall/lambda_type * self.Cy_south[idx, :]

        elif outer_wall == "west":
            M = (self.X == self.x[-1]) & (self.Y >= int(start / hx)) & (self.Y <= int(end / hx))
            idx = M.flatten()

            self.A[idx, :] = lambda_wall/lambda_type * self.Cx_west[idx, :]


    def heat_up(self, T_total): #całkowity czas podawany w godzinach
        u_current = self.current_temp.copy()
        u_current = u_current.flatten()
        T = int(T_total*3600)

        for _ in range(int(T/ht)):
            u_current[self.idx_north_wall] = self.north_out_temp
            u_current[self.idx_east_wall] = self.east_out_temp
            u_current[self.idx_south_wall] = self.south_out_temp
            u_current[self.idx_west_wall] = self.west_out_temp


            u_current += self.heat_sources * (self.radiators_heat_level / 5)


            u_current = np.linalg.solve(self.A, u_current)

        self.current_temp = u_current.reshape(self.Ny, self.Nx)



    def temp_plot(self):
        u_celc = self.current_temp - 273.15

        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        z_min = 15
        z_max = 25
        levels = np.linspace(z_min, z_max, 44)

        im1 = axs.contourf(self.X, self.Y, u_celc, levels=levels, cmap='viridis')
        fig.colorbar(im1, ax=axs, ticks=np.linspace(z_min, z_max, 6))

        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()



if __name__ == "__main__":
    moje_mieszkanie = Flat(6, 10, 22, 18,
                           21, 21, 21, 21)

    moje_mieszkanie.add_outer_object("south", 2, 6, lambda_window)

    moje_mieszkanie.add_outer_object("north", 4.6, 5.6, lambda_door)

    moje_mieszkanie.add_radiator("horizontal", 0.1,
                                 0.2, 1, 9.6)
    moje_mieszkanie.add_radiator("vertical", 0.1,
                                 3.2, 3.8, 3.8)

    moje_mieszkanie.add_inner_object("horizontal", 4,
                                     0.2, 4, lambda_wall)
    moje_mieszkanie.add_inner_object("vertical", 4,
                                     0.2, 4, lambda_wall)
    moje_mieszkanie.add_inner_object("vertical", 4,
                                     1, 2, lambda_door)

    moje_mieszkanie.change_radiators_heat_level(5)

    time_2 = time()


    moje_mieszkanie.heat_up(1)
    time_3 = time()
    print(time_3 - time_2)

    moje_mieszkanie.temp_plot()
