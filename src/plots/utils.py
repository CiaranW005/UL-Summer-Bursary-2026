import numpy as np

def sphere_surface(
        center: tuple[float, float, float], 
        radius: float, 
        resolution: int =20
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the surface cordinates of a sphere"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    return x, y, z