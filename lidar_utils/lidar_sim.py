import math
import random
import time

SIM_QUALITY = 15  # Simulated quality value passed to on_update

def read_lidar_data(on_update, width=1.0, length=2.0, step_size=1, proximity=0.1):
    """
    Simulates RPLidar C1 sensor data continuously.
    Calls on_update(angle, distance, quality) for each measurement.
    Distances are in millimeters, matching the real sensor format.
    Stops on KeyboardInterrupt.
    """
    print("Starting simulated scan...")
    _, _, _, _, _, _, results = get_boundary_distances(width, length, step_size, proximity)
    scan = [(angle, int(dist_m * 1000)) for angle, dist_m in results if 0 < int(dist_m * 1000) < 12000]
    try:
        while True:
            for angle, distance_mm in scan:
                on_update(angle, distance_mm, SIM_QUALITY)
            time.sleep(0.1)  # Pause between scan cycles
    except KeyboardInterrupt:
        print("\nStopping simulation...")


def get_boundary_distances(width=1.0, length=2.0, step_size=1, proximity=0.1):
    # 1. Random source point and orientation
    px, py = random.uniform(0, width), random.uniform(0, length)
    angle_f = random.uniform(0, 360)
    
    # 2. Generate 3 random obstacle points
    obstacles = []
    for _ in range(3):
        ox = random.uniform(0.1, width - 0.1)
        oy = random.uniform(0.1, length - 0.1)
        obstacles.append((ox, oy))
    
    results = []
    
    for angle_deg in range(0, 360, step_size):
        angle_rad = math.radians(angle_deg + angle_f)
        ux = math.cos(angle_rad) # Unit vector X
        uy = math.sin(angle_rad) # Unit vector Y
        
        # --- Wall Boundary Calculation ---
        dist_x_max = (width - px) / ux if ux > 0 else float('inf')
        dist_x_min = (0 - px) / ux if ux < 0 else float('inf')
        dist_y_max = (length - py) / uy if uy > 0 else float('inf')
        dist_y_min = (0 - py) / uy if uy < 0 else float('inf')
        
        dist_to_wall = min(dist_x_max, dist_x_min, dist_y_max, dist_y_min)
        
        # --- Obstacle Detection ---
        min_dist = dist_to_wall
        
        for ox, oy in obstacles:
            # Vector from source to obstacle
            vx, vy = ox - px, oy - py
            
            # Distance along the ray (Projection)
            d_proj = vx * ux + vy * uy
            
            if d_proj > 0: # Obstacle is in front of the ray
                # Perpendicular distance from obstacle to ray
                d_perp = abs(vx * uy - vy * ux)
                
                if d_perp < proximity:
                    # Calculate distance to the edge of the circular proximity zone
                    # Using Pythagorean theorem: dist^2 + d_perp^2 = proximity^2
                    hit_dist = d_proj - math.sqrt(proximity**2 - d_perp**2)
                    
                    if 0 < hit_dist < min_dist:
                        min_dist = hit_dist
        
        results.append((angle_deg, min_dist))
        
    return px, py, angle_f, width, length, obstacles, results
