import numpy as np

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