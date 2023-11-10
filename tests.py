"""
Calificación del laboratorio
-----------------------------------------------------------------------------------------
"""

import sys

import numpy as np

import preguntas



def test_01():
    """
    ---< Run command >-------------------------------------------------------------------
    Pregunta 01
    pip3 install scikit-learn pandas numpy
    python3 tests.py 01
    """

    x_poly, _ = preguntas.pregunta_01()
    x_poly = x_poly.round(3)
    x_expected = np.array(
        [
            [1.0, -4.0, 16.0],
            [1.0, -3.579, 12.809],
            [1.0, -3.158, 9.972],
            [1.0, -2.737, 7.49],
            [1.0, -2.316, 5.363],
            [1.0, -1.895, 3.59],
            [1.0, -1.474, 2.172],
            [1.0, -1.053, 1.108],
            [1.0, -0.632, 0.399],
            [1.0, -0.21, 0.044],
            [1.0, 0.21, 0.044],
            [1.0, 0.632, 0.399],
            [1.0, 1.053, 1.108],
            [1.0, 1.474, 2.172],
            [1.0, 1.895, 3.59],
            [1.0, 2.316, 5.363],
            [1.0, 2.737, 7.49],
            [1.0, 3.158, 9.972],
            [1.0, 3.579, 12.809],
            [1.0, 4.0, 16.0],
        ]
    )

    for i in range(x_poly.shape[0]):
        for j in range(x_poly.shape[1]):
            assert x_poly[i, j] == x_expected[i, j]


def test_02():
    """
    ---< Run command >-------------------------------------------------------------------
    Pregunta 02
    pip3 install scikit-learn pandas numpy
    python3 tests.py 02
    """
    params = preguntas.pregunta_02()
    expected = np.array([0.666, -3.0, 2.032])
    assert np.allclose(params, expected, atol=1e-3)


test = {
    "01": test_01,
    "02": test_02,
}[sys.argv[1]]

test()
