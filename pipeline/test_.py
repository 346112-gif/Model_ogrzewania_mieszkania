import numpy as np
import scipy.sparse as sp
from scipy.sparse import kron, eye, csr_matrix

x = np.arange(0, 8, 1)
y = np.arange(0, 4, 1)

X, Y = np.meshgrid(x, y)

print(X[1:-1, 1:-1])
print(X)


def add_inner_object(self, radiator_placement, dist_from_wall, start, end, lambda_type):
    if radiator_placement == "north":
        M = np.where((self.Y == self.y[int(dist_from_wall / hx)])
                     & (self.X >= start) & (self.X <= end), True, False)
        idx = M.flatten()
        self.A[idx, :] = (air_heat_coeff / lambda_type *
                          np.kron(D1_backward(self.Ny), self.id_Nx) / hy + self.id_Nxy)[idx, :]

    elif radiator_placement == "east":
        M = np.where((self.X == self.x[int(dist_from_wall / hx)])
                     & (self.Y >= start) & (self.Y <= end), True, False)
        idx = M.flatten()
        self.A[idx, :] = (-air_heat_coeff / lambda_type *
                          np.kron(self.id_Ny, D1_forward(self.Nx)) / hx + self.id_Nxy)[idx, :]

    elif radiator_placement == "south":
        M = np.where((self.Y == self.y[int(dist_from_wall / hx)])
                     & (self.X >= start) & (self.X <= end), True, False)
        idx = M.flatten()

        self.A[idx, :] = (-air_heat_coeff / lambda_type *
                          np.kron(D1_forward(self.Ny), self.id_Nx) / hy + self.id_Nxy)[idx, :]

    elif radiator_placement == "west":
        M = np.where((self.X == self.x[int(dist_from_wall / hx)])
                     & (self.Y >= start) & (self.Y <= end), True, False)
        idx = M.flatten()

        self.A[idx, :] = (air_heat_coeff / lambda_type *
                          np.kron(self.id_Ny, D1_backward(self.Nx)) / hx + self.id_Nxy)[idx, :]



    def add_inner_object(self, placement, dist_from_wall, start, end, lambda_type):
        ratio = lambda_type / lambda_air

        if placement == "vertical":
            M = (self.X == self.x[int(dist_from_wall / hx)]) & (self.Y >= start) & (self.Y <= end)
        elif placement == "horizontal":
            M = (self.Y == self.y[int(dist_from_wall / hy)]) & (self.X >= start) & (self.X <= end)

        idx = M.flatten()

        self.A[idx, :] = self.id_Nxy[idx, :] - (self.alfa * ratio * ht * self.laplacian)[idx, :]



if __name__ == "__main__":
    moje_mieszkanie = Flat(6, 10, 2.5, 22, 22,
                           19, 22, -5, 22)

    moje_mieszkanie.add_outer_object("south", 2.6, 4.8, lambda_window)

    moje_mieszkanie.add_outer_object("north", 4.6, 5.6, lambda_door)

    moje_mieszkanie.add_radiator("horizontal", hx, 0.8,
                                 1, 2.6, 9.8)
    moje_mieszkanie.add_radiator("vertical", hx, 0.8,
                                 3, 3.8, 3.8)

    moje_mieszkanie.add_inner_object("horizontal", 4, 0.3,
                                     0.2, 4, lambda_wall)
    moje_mieszkanie.add_inner_object("vertical", 4, 0.2,
                                     0.2, 4, lambda_wall)
    moje_mieszkanie.add_inner_object("vertical", 4, 0.3,
                                     1, 2, lambda_door)

    #moje_mieszkanie.turn_heating_on()
    time1 = time()
    #moje_mieszkanie.heat_up(8)
    moje_mieszkanie.change_radiators_heat_level(0)
    moje_mieszkanie.heat_up(8)

    moje_mieszkanie.temp_plot()

    moje_mieszkanie.change_radiators_heat_level(5)
    moje_mieszkanie.heat_up(8)
    time2 = time()
    print(time2 - time1)

    moje_mieszkanie.temp_plot()

