"""Funciones base utilizadas para trazar geometrías fotovoltaicas 2D"""

def plot_coords(ax, ob):
    """Trazar coordenadas de objetos con formas

     Parámetros
     ----------
     ax: objeto ``matplotlib.pyplot.Axes``
         Ejes para trazar
     ob: objeto ``bien formado``
         Objeto geométrico cuyas coordenadas x,y deben trazarse

     """
    try:
        x, y = ob.xy
        ax.plot(x, y, 'o', color='#999999', zorder=1)
    except NotImplementedError:
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)


def plot_bounds(ax, ob):
    """Trazar los límites de un objeto con formas

     Parámetros
     ----------
     ax: objeto ``matplotlib.pyplot.Axes``
         Ejes para trazar
     ob: objeto ``bien formado``
         Objeto geométrico cuyos límites se deben trazar

     """
    # Check if shadow reduces to one point (for very specific sun alignment)
    if len(ob.boundary) == 0:
        x, y = ob.coords[0]
    else:
        x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, 'o', color='#000000', zorder=1)


def plot_line(ax, ob, line_color):
    """Trazar los límites de una línea bien proporcionada

     Parámetros
     ----------
     ax: objeto ``matplotlib.pyplot.Axes``
         Ejes para trazar
     ob: objeto ``bien formado``
         Objeto geométrico cuyos límites se deben trazar
     line_color: cadena
         color de matplotlib a usar para trazar la línea

     """
    try:
        x, y = ob.xy
        ax.plot(x, y, color=line_color, alpha=0.7,
                linewidth=3, solid_capstyle='round', zorder=2)
    except NotImplementedError:
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, color=line_color,
                    alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)