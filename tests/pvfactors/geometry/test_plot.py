import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point
import pytest
from  xm_solarlib.pvfactors.geometry.plot import plot_bounds, plot_coords, plot_line

# Define objetos para las pruebas
point = Point(0, 0)
line = LineString([(0, 0), (1, 1), (2, 0)])
multi_line = MultiLineString([[(0, 0), (1, 1), (2, 0)], [(0, 2), (1, 1), (2, 2)]])
polygon = Polygon([(0, 0), (1, 1), (1, 0)])
multi_polygon = MultiPolygon([Polygon([(0, 0), (1, 1), (1, 0)]), Polygon([(2, 2), (3, 3), (3, 2)])])

@pytest.fixture
def setup_axes():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)

def test_plot_coords_point(setup_axes):
    ax = setup_axes
    plot_coords(ax, point)
    # Asserts para las coordenadas del punto
    assert len(ax.lines) == 1  # Debería haber un solo punto trazado

def test_plot_line(setup_axes):
    ax = setup_axes
    line_color = 'red'
    plot_line(ax, line, line_color)
    # Asserts para la línea
    assert len(ax.lines) == 1  # Se traza una sola línea

def test_plot_line_multi_line(setup_axes):
    ax = setup_axes
    line_color = 'blue'
    plot_line(ax, multi_line, line_color)
    # Asserts para la línea múltiple
    assert len(ax.lines) == 2  # Se trazan dos líneas