import numpy as np

def simple_corners(points):
	"""
    Takes the list of boundary points.
    Returns a list of (x, y) coordinates where points are further away than their neighbors and close enough to them.
    """
	corners = []
	i = 1
	while i < len(points) -1:
		d = (points[i][0]**2 + points[i][1]**2) ** 0.5
		# Depth check
		if d > (points[i - 1][0]**2 + points[i - 1][1]**2) ** 0.5 and d > (points[i + 1][0]**2 + points[i + 1][1]**2) ** 0.5:
			# Distance check
			if ((points[i][0] - points[i - 1][0])**2 + (points[i][1] - points[i - 1][1])**2) ** 0.5 < 0.1 and ((points[i][0] - points[i + 1][0])**2 + (points[i][1] - points[i + 1][1])**2) ** 0.5 < 0.1:
				corners.append(points[i])
		i += 1
	return corners
