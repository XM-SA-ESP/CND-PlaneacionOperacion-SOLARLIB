from  xm_solarlib.pvfactors.geometry.utils import difference, contains
from  xm_solarlib.pvfactors.geometry.base import BaseSide
from shapely.geometry import Point, LineString, MultiLineString


def test_difference():
    """Testing own implementation of geometry difference operator"""

    # Simple cases
    u = LineString([(0, 0), (2, 0)])

    v = LineString([(1, 0), (3, 0)])
    diff = difference(u, v)
    assert diff == LineString([(0, 0), (1, 0)])

    v = LineString([(3, 0), (1, 0)])
    diff = difference(u, v)
    assert diff == LineString([(0, 0), (1, 0)])

    v = LineString([(-1, 0), (1, 0)])
    diff = difference(u, v)
    assert diff == LineString([(1, 0), (2, 0)])

    v = LineString([(1, 0), (-1, 0)])
    diff = difference(u, v)
    assert diff == LineString([(1, 0), (2, 0)])

    v = LineString([(0.5, 0), (1.5, 0)])
    diff = difference(u, v)
    assert diff == MultiLineString([((0, 0), (0.5, 0)), ((1.5, 0), (2, 0))])

    v = LineString([(1.5, 0), (0.5, 0)])
    diff = difference(u, v)
    assert diff == MultiLineString([((0, 0), (0.5, 0)), ((1.5, 0), (2, 0))])

    v = LineString([(1, 0), (1, 1)])
    diff = difference(u, v)
    assert diff == u

    v = LineString([(1, 1), (1, 0)])
    diff = difference(u, v)
    assert diff == u

    v = LineString([(1, 1), (1, 2)])
    diff = difference(u, v)
    assert diff == u

    v = LineString([(0, 0), (1, 0)])
    diff = difference(u, v)
    assert diff == LineString([(1, 0), (2, 0)])

    # Case with potentially float error
    u = LineString([(0, 0), (3, 2)])
    v = LineString([(0, 0), u.interpolate(0.5, normalized=True)])
    diff = difference(u, v)
    assert diff.length == u.length / 2.

    # Case were should return empty geoemtry
    diff = difference(u, u)
    assert isinstance(diff, LineString)
    assert diff.is_empty

    # Special case that caused crash
    u = LineString([(1, 0), (0, 0)])
    v = LineString([(0, 0), (2, 0)])
    diff = difference(u, v)
    assert diff.is_empty

    # Special case that caused crash
    u = LineString([(1, 0), (0, 0)])
    v = LineString([(-2, 0), (1, 0)])
    diff = difference(u, v)
    assert diff.is_empty


def test_contains_on_side():
    """Check that ``contains`` function works on a BaseSide instance"""
    coords = [(0, 0), (2, 0)]
    side = BaseSide.from_linestring_coords(coords)
    point = Point(1, 0)
    assert contains(side, point)