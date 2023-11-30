"""
Optimización usando gradiente descendente - Regresión polinomial
-----------------------------------------------------------------------------------------

En este laboratio se estimarán los parámetros óptimos de un modelo de regresión 
polinomial de grado `n`.

"""


def pregunta_01():
    """
    Complete el código presentado a continuación.
    """
    # Importe pandas
    import pandas as pd

    # Importe PolynomialFeatures
    from sklearn.preprocessing import PolynomialFeatures

    # Cargue el dataset `data.csv`
    data = pd.read_csv("data.csv")

    # Cree un objeto de tipo `PolynomialFeatures` con grado `2`
    poly = PolynomialFeatures(degree=2)

    # Transforme la columna `x` del dataset `data` usando el objeto `poly`
    x_poly = poly.fit_transform(data[["x"]])

    # Retorne x y y
    return x_poly, data.y


def pregunta_02():

    # Importe numpy
    import numpy as np

    x_poly, y = pregunta_01()

    # Fije la tasa de aprendizaje en 0.0001 y el número de iteraciones en 1000
    learning_rate = 0.0001
    n_iterations = 1000

    # Defina el parámetro inicial `params` como un arreglo de tamaño 3 con ceros
    params = np.zeros(3)
    for _ in range(n_iterations):

        # Compute el pronóstico con los parámetros actuales
        y_pred = np.dot(x_poly, params)

        # Calcule el error
        error = y_pred - y

        # Calcule el gradiente
        gradient = np.dot(x_poly.T, error)

        # Actualice los parámetros
        params = params - learning_rate * gradient

    return params
