import numpy as np
from xm_solarlib.pvfactors.config import TOL_COLLINEAR, DISTANCE_TOLERANCE
from shapely.geometry import LineString, MultiLineString


def _difference_if_u_contains_vb2(u, ub1, ub2, vb2, v_contains_ub1, v_contains_ub2):
    if v_contains_ub1:
        if v_contains_ub2:
            return LineString()
        else:
            return LineString([vb2, ub2])
    elif v_contains_ub2:
        return LineString([ub1, vb2])
    else:
        return u

def _difference_if_u_contains_vb1(u, ub1, ub2, vb1, vb2, u_contains_vb2,v_contains_ub1, v_contains_ub2 ):
    if u_contains_vb2:
        l_tmp = LineString([ub1, vb1])
        if contains(l_tmp, vb2):
            list_lines = [LineString([ub1, vb2]), LineString([vb1, ub2])]
        else:
            list_lines = [LineString([ub1, vb1]), LineString([vb2, ub2])]
        # Note that boundary points can be equal, so need to make sure
        # we're not passing line strings with length 0
        final_list_lines = [line for line in list_lines
                            if line.length > DISTANCE_TOLERANCE]
        len_final_list = len(final_list_lines)
        if len_final_list == 2:
            return MultiLineString(final_list_lines)
        elif len_final_list == 1:
            return final_list_lines[0]
        else:
            return LineString()
    elif v_contains_ub1:
        if v_contains_ub2:
            return LineString()
        else:
            return LineString([vb1, ub2])
    elif v_contains_ub2:
        return LineString([ub1, vb1])
    else:
        return u
    
def difference(u, v):
    """Calcular la diferencia entre dos líneas, evitando flotar bien proporcionado
     errores de precisión

     Parámetros
     ----------
     u: :py:clase:`shapely.geometry.LineString`-como
         Cadena de línea de la cual se eliminará ``v``
     v : :py:clase:`shapely.geometry.LineString`-like
         Cadena de línea para eliminar de ``u``

     Devoluciones
     -------
     :py:clase:`shapely.geometry.LineString`
        Diferencia resultante de la superficie actual menos la cadena lineal dada
     """
    ub1, ub2 = u.boundary
    vb1, vb2 = v.boundary
    u_contains_vb1 = contains(u, vb1)
    u_contains_vb2 = contains(u, vb2)
    v_contains_ub1 = contains(v, ub1)
    v_contains_ub2 = contains(v, ub2)

    if u_contains_vb1:
        return _difference_if_u_contains_vb1(u, ub1, ub2, vb1, vb2, u_contains_vb2,v_contains_ub1, v_contains_ub2 )
    elif u_contains_vb2:
        return _difference_if_u_contains_vb2(u, ub1, ub2, vb2, v_contains_ub1, v_contains_ub2)
    else:
        return u
 

def contains(linestring, point, tol_distance=DISTANCE_TOLERANCE):
    """Corregir errores de coma flotante obtenidos en Shapely for contiene"""
    return linestring.distance(point) < tol_distance


def is_collinear(list_elements):
    """Compruebe que todo :py:class:`~pvfactors.pvsurface.PVSegment`
     o :py:class:`~pvfactors.pvsurface.PVSurface` objetos en la lista
     son colineales"""
    is_col = True
    u_direction = None  # will be orthogonal to normal vector
    for element in list_elements:
        if not element.is_empty:
            if u_direction is None:
                # set u_direction if not defined already
                u_direction = np.array([- element.n_vector[1],
                                        element.n_vector[0]])
            else:
                # check that collinear
                dot_prod = u_direction.dot(element.n_vector)
                is_col = np.abs(dot_prod) < TOL_COLLINEAR
                if not is_col:
                    return is_col
    return is_col


def check_collinear(list_elements):
    """Aparece un error si todo :py:class:`~pvfactors.pvsurface.PVSegment`
     o :py:class:`~pvfactors.pvsurface.PVSurface` objetos en la lista
     no son colineales"""
    is_col = is_collinear(list_elements)
    if not is_col:
        msg = "All elements should be collinear"
        raise ValueError(msg)
    

def are_2d_vecs_collinear(u1, u2):
    """Comprueba que dos vectores 2D son colineales"""
    n1 = np.array([-u1[1], u1[0]])
    dot_prod = n1.dot(u2)
    return np.abs(dot_prod) < TOL_COLLINEAR