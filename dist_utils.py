import numpy as np
import math

# Global definitions (assumed defined elsewhere)
grass_light = np.array([100, 228, 100])
grass_dark  = np.array([100, 202, 100])
grass = (grass_dark+ grass_light)/2
road        = np.array([100, 100, 100])
toolbar     = np.array([0, 0, 0])
car_color   = np.array([192, 15, 15])

size_x = 96
size_y = 96
car_x = 48   # x position (from left)
car_y = 73   # y position (from top)

# Assume 'image' is a NumPy array of shape (size_y, size_x, 3)
# representing the background (without the car drawn on top)

def approx_equal(a, b, eps = 30):
    return np.sum(np.abs(a-b)) < eps

def get_category(pixel):
    """
    Given an RGB pixel (as a numpy array), return its category.
    Returns:
       "grass" if the pixel matches either grass_light or grass_dark,
       "road"  if it equals road,
       "toolbar" if it equals toolbar,
       "car" if it equals car_color,
       None otherwise.
    """
    if approx_equal(pixel, grass_light):
        return "grass"
    elif approx_equal(pixel, road):
        return "road"
    elif approx_equal(pixel, toolbar):
        return "toolbar"
    elif approx_equal(pixel, car_color):
        return "car"
    else:
        return None

def compute_max_distance(x0, y0, dx, dy):
    """
    Compute the maximum distance we can travel along the ray starting at (x0, y0)
    with direction (dx, dy) before leaving the image bounds.
    """
    t_x = float('inf')
    t_y = float('inf')
    # x coordinate: image valid if 0 <= x < size_x
    if dx > 0:
        t_x = (size_x - 1 - x0) / dx
    elif dx < 0:
        t_x = (0 - x0) / dx
    # y coordinate: image valid if 0 <= y < size_y
    if dy > 0:
        t_y = (size_y - 1 - y0) / dy
    elif dy < 0:
        t_y = (0 - y0) / dy
    return min(t_x, t_y)

def get_distance(image, theta):
    """
    Cast a ray from (car_x, car_y) in direction theta (radians, where 0 is up)
    until you hit one of the background surfaces (grass, road, or toolbar).
    
    If the first background pixel encountered is:
       - road: then continue until a pixel that is not road (i.e. grass or toolbar)
         is found. Return a normalized value in [0, 1] where 0 means the boundary
         is immediate and 1 means no grass/toolbar was found.
       - grass: then continue until a pixel that is not grass (i.e. road or toolbar)
         is found. Return a normalized value in [-1, 0] where 0 means the boundary is
         immediate and -1 means no road/toolbar was found.
         
    The normalization is done relative to the distance (from just after leaving the car)
    to the image boundary in that direction.
    """
    # Convert theta to a unit vector.
    # With 0 pointing up and y increasing downward, the unit vector is:
    dx = math.sin(theta)
    dy = -math.cos(theta)
    
    # A small step size (in pixels) for the ray-cast.
    step = 0.5
    x0, y0 = car_x, car_y

    # Compute how far we can go before exiting the image.
    t_max = compute_max_distance(x0, y0, dx, dy)
    
    base_category = None  # This will be "road" or "grass"
    d_base = None         # Distance from starting point where we first see a background pixel
    d = 0
    boundary_found = False
    d_boundary = None

    while d <= t_max:
        # Compute current (floating-point) position along the ray.
        x_sample = x0 + d * dx
        y_sample = y0 + d * dy
        # Convert to nearest pixel indices
        x_int = int(round(x_sample))
        y_int = int(round(y_sample))
        
        # Check if we're inside the image
        if x_int < 0 or x_int >= size_x or y_int < 0 or y_int >= size_y:
            break
        
        pixel = image[y_int, x_int]  # image is indexed as [row, column]
        cat = get_category(pixel)
        
        # Skip if we hit the car (or an undefined pixel)
        if cat is None or cat == "car":
            d += step
            continue
        
        # For the very first background pixel we see...
        if base_category is None:
            # If the first encountered pixel is toolbar (unusual) treat that as an immediate boundary.
            if cat == "toolbar":
                return 0
            else:
                base_category = cat  # either "road" or "grass"
                d_base = d
        else:
            # Now look for the transition: if the current pixel’s category is different,
            # then we have found the boundary.
            if cat != base_category:
                d_boundary = d
                boundary_found = True
                break
        d += step

    # If no boundary was found before leaving the image, we assume the ray remains on one surface.
    if not boundary_found:
        if base_category == "road":
            return 1
        elif base_category == "grass":
            return -1
        else:
            # In case no background pixel was ever encountered,
            # return 0 as a fallback.
            return 0
    else:
        # Normalize the distance from where the background started (d_base) to the boundary (d_boundary)
        # relative to the available distance (from d_base to t_max).
        normalized = (d_boundary - d_base) / (t_max - d_base)
        if base_category == "road":
            # Positive: 0 means boundary immediately; 1 means no grass/toolbar encountered.
            return normalized
        elif base_category == "grass":
            # Negative: 0 means boundary immediately; -1 means no road/toolbar encountered.
            return -normalized

# Example usage:
# theta = 0          # straight up
# result = get_distance(theta)
# print("Distance measure:", result)
