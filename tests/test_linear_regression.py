import pytest
import numpy as np
from linear_regression import LinearRegression

@pytest.fixture
def data() -> float|int:
    return np.array([[2, 4], [3, 5], [4, 6], [5, 7]]), np.array([10, 30, 40, 50])

def test_fit():
    # test coeff are correctly initialized
    assert LinearRegression().intercept == 0
    assert LinearRegression().coef == 0

def test_fit_with_data(data):
    X, y = data

    my_regression = LinearRegression()

    my_regression.fit(X, y)
    assert my_regression.coef.__round__(2) == 19.5
    assert my_regression.intercept.__round__(2) == -6.5

def test_predict(data):
    X, y = data

    my_regression = LinearRegression()

    my_regression.fit(X, y)

    assert [x.__round__(0) for x in my_regression.predict(np.array([[18, 22], [19, 23], [20, 24], [21, 25]]))] == [208, 221, 234, 247]
