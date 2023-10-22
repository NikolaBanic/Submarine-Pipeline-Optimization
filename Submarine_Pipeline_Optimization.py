import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import *
from matplotlib import cm
from scipy.interpolate import griddata
import math
import rasterio
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import LineString
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

"""
Loading the .xyz coordinates data 
"""

data = pd.read_csv('Input_Data/region.csv')

x_d = data['x'].tolist()
y_d = data['y'].tolist()
z_d = data['z'].tolist()

xi = np.linspace(min(x_d), max(x_d), 200)
yi = np.linspace(min(y_d), max(y_d), 200)

X, Y = np.meshgrid(xi, yi)
Z = griddata((x_d,y_d), z_d, (X, Y), method = 'linear')

x_min, x_max = np.min(X), np.max(X)
y_min, y_max = np.min(Y), np.max(Y)
z_min, z_max = np.nanmin(Z), np.nanmax(Z)

x = X[0,:]
y = Y[:,0]

"""
Path of prepared shapefiles
"""
folder1 = 'Input_Data/sea_floor_type'
folder2 = 'Input_Data/constraints'
names1 = ['yellow','dark_y','orange','red', 'blue'] 
names2 = ['boreholes', 'fishing', 'corals', 'ship_wrecks', 'submarine_landslides', 'wind_farms']
path1 = [f'{folder1}/yellow.shp', f'{folder1}/dark_y.shp', f'{folder1}/orange.shp', f'{folder1}/red.shp', f'{folder1}/blue.shp']
path2 = [f'{folder2}/boreholes.shp', f'{folder2}/fishing.shp', f'{folder2}/corals.shp', f'{folder2}/ship_wrecks.shp', f'{folder2}/submarine_landslides.shp', f'{folder2}/wind_farms.shp']

"""
Loading shapefiles for sea floor types
"""
# Ranked from best to worst 
yellow = gpd.read_file(f'{path1[0]}') # sand <= best
dark_y = gpd.read_file(f'{path1[1]}') # 
orange = gpd.read_file(f'{path1[2]}') # 
blue = gpd.read_file(f'{path1[4]}') #
red = gpd.read_file(f'{path1[3]}') # rocks <= worst
"""
Loading shapefiles for constraints
"""
boreholes = gpd.read_file(f'{path2[0]}')
fishing = gpd.read_file(f'{path2[1]}')
corals = gpd.read_file(f'{path2[2]}')
ship_wrecks = gpd.read_file(f'{path2[3]}')
submarine_landslides = gpd.read_file(f'{path2[4]}')
wind_farms = gpd.read_file(f'{path2[5]}')

fileovi1 = [yellow, dark_y, orange, red, blue]
fileovi2 = [boreholes, fishing, corals, ship_wrecks, submarine_landslides, wind_farms]

def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth's radius in kilometers

    # Convert latitude and longitude from decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate differences in latitude and longitude
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 *np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c

    # Convert distance from kilometers to meters
    distance_meters = distance_km * 1000

    return distance_meters

def is_line_inside_shapefile(line_coords, shapefile):
    # Create a LineString object from the line coordinates
    line = LineString(line_coords)


    # Check if the entire line intersects with any of the polygons in the shapefile
    for idx, polygon in shapefile.iterrows():
        if line.intersects(polygon['geometry']):
            return 1

    return 0

def bilinear_interpolation(point):
    p_x, p_y = point
    if p_x >= x_min and p_x <= x_max and p_y >= y_min and p_y <= y_max:

        i_x = np.argwhere(p_x <= x)[0]
        i_y = np.argwhere(p_y <= y)[0]
        i_x, i_y = i_x[0], i_y[0]
        
        # Interpolation points
        x1, x2 = x[i_x - 1], x[i_x]
        y1, y2 = y[i_y], y[i_y - 1]
        z1, z2 = Z[i_y - 1, i_x - 1], Z[i_y - 1, i_x]
        z3, z4 = Z[i_y, i_x - 1] , Z[i_y, i_x]
        
        # z1 upper left, z2 upper right, z3 lower left, z4 lower right
        p_z = (z3 * (x2 - p_x) * (y2 - p_y) + z4 * (p_x - x1) * (y2 - p_y) \
            + z1 * (x2 - p_x) * (p_y - y1) + z2 * (p_x - x1) * (p_y - y1)) / ((x2 - x1) * (y2 - y1)) 
        constraint = 0
        # If point is outsie the area give it random height and constraint is not satisifed
        if np.isnan(p_z):
            p_z = np.random.uniform(z_max/z_min, 2*z_max/z_min)
            constraint = 1
    else:
        p_z = np.random.uniform(z_max/z_min, 2*z_max/z_min)
        constraint = 1

    return p_z, constraint

def random_path(theta, l, A):
    # Generates random coordinates for pipeline route
    
    n = np.size(theta)  # Number of pipes
    phi = np.zeros(n)  
    phi[0] = theta[0]
    

    # Coordinates of starting route
    x_ = np.zeros(n + 1)
    y_ = np.zeros(n + 1)
    
    # Starting points
    x_[0] = A[0]
    y_[0] = A[1]

    # Angles
    for i in range(1, n):
        phi[i] = phi[i-1] + theta[i]


    # Coordinates
    for i in range(1, n + 1):
        x_[i] = x_[i - 1] + np.cos(phi[i - 1]) * l
        y_[i] = y_[i - 1] + np.sin(phi[i - 1]) * l

    return x_, y_
    
def distance(x, y, z):
    L = np.empty(len(x)-1)
    slope = np.empty(len(x)-1)

    for i in range(len(x) - 1):
        L_ = (haversine(x[i],y[i],x[i+1],y[i+1]))

        H_ = np.abs(z[i+1] - z[i])

        L[i] = np.sqrt(L_**2 + H_**2)
  
        slope[i] = np.degrees(np.arctan2(H_, L_))
       
        
    print(f'Lenght of the route: {np.sum(L)/1000:.2f} km')
    return L, slope

def exploatation(L, z):
    Q = 0.0092 * 5 # m3/s # 5000 Barrel * 5
    D = 0.3 # m
    rho = 755 # kg/m3
    rho_w = 1000 # kg/m3
    eps = 0.05 * 1e-3 # m
    nu = 1e-5 # m2/s
    T = 25 * 365 * 24 * 60 * 60 # s
    g = 9.81 # m/s2
    eta = 0.85
    price_el = 0.16  # €/kWh
    price_el = price_el * 1e-3 / 3600  # €/Ws
    
    # Brzina strujanja
    v = Q / (D**2 * np.pi * 0.25)
    # Reynolds number
    Re = (v * D) / nu
    # Faktor trenja
    f =  0.11 * ((eps/D) + (68/Re))** 0.25

    # print(v,Re,f)
        
    h_l = np.empty(len(L))
    h_P = np.empty(len(L))
    
    for i in range(len(L)):
        h_l[i] = f * (L[i]/D) * (v**2/2*g)
        p_A, p_B = abs(rho_w * g * z[i+1]), abs(rho_w * g * z[i])
        h_P[i] = p_B/(rho*g) - p_A/(rho*g) + abs(z[i+1]) - abs(z[i]) + h_l[i]
    
    P =  Q * rho * g * np.sum(h_P) * eta
    
    price_EXP =  P * T * price_el / 1000000
    
    return price_EXP, P


def path(theta, plot=False, plot_constraint=False):
    
    # Function that performs the main path calculation and visualization
    # print(f'theta = {theta}')

    A = [Ax, Ay]
    B = [Bx, By]
    
    # Initial unit path
    # Append the absolute angle of the first segment to the beginning of the theta vector
    theta1 = np.hstack([0, theta])  

    scale = 1
    x1, y1 = random_path(theta1, scale, A)

    # Rotation and scaling
    phi_ab = np.arctan2(B[1] - A[1], B[0] - A[0])  # Angle of line AB
    # print(np.rad2deg(phi_ab))
    
    phi_ab1 = np.arctan2(y1[-1] - A[1], x1[-1] - A[0])  # Angle of line AB''
    # print(np.rad2deg(phi_ab1))
    
    l_ab = np.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)

    l_ab1 = np.sqrt((x1[-1] - x1[0])**2 + (y1[-1] - y1[0])**2)

    theta2 = np.copy(theta1)
    theta2[0] = phi_ab - phi_ab1
    l2 = l_ab / l_ab1

    x2, y2 = random_path(theta2, l2 * scale, A)
    z2 = np.zeros_like(x2) 
    
    factors = np.zeros((len(x2), len(names1)+1))
    constraints = np.zeros((len(x2), len(names2)+1))
    f0, f1, f2, f3, f4 = 0, 0.1, 0.2, 1.2, 0.3
    
    # Sediment factors
    for j in range(len(fileovi1)):
       for i in range(len(x2)-1):  
           factors[i, j] = is_line_inside_shapefile([(x2[i],y2[i]),(x2[i+1],y2[i+1])], fileovi1[j])
           
    # Elevation of all pipeline points
    for i in range(len(x2)):
       z2[i] = bilinear_interpolation([x2[i], y2[i]])[0]

    # Constraints
    for j in range(len(fileovi2)):
       for i in range(len(x2)-1):
           constraints[i, 0] = bilinear_interpolation([x2[i], y2[i]])[1]
           constraints[i, j+1] = is_line_inside_shapefile([(x2[i],y2[i]),(x2[i+1],y2[i+1])], fileovi2[j])
           # if  constraints[i, 0] == 1:
           #     print('Outside the map')
           # if  constraints[i, 1] == 1:
           #     print('Boreholes')
           # if  constraints[i, 2] == 1:
           #     print('Holes in the coast and islands')
           # if  constraints[i, 3] == 1:
           #     print('Protected area')
           # if  constraints[i, 4] == 1:
           #     print('Shipwrecks')
           # if  constraints[i, 5] == 1:
           #     print('Landslide area')
           # if constraints[i, 6] == 1:
           #     print('Wind turbines')
               
    constraint = np.sum(constraints)    
       
    L, S = distance(x2, y2, z2)
    L_SUM = np.sum(L)
    """
    Installation cost
    """
    factors[:,0] *= f0
    factors[:,1] *= f1
    factors[:,2] *= f2
    factors[:,3] *= f3
    factors[:,4] *= f4
    
    sediment_list = np.max(factors[:,:4], axis = 1)
    sediment_list[sediment_list == 0] = f4
    
    slope_list = [0 if slope < 2 else
                   0.3 if 2 <= slope < 5 else
                   0.9 if 5 <= slope < 10 else
                   1.8 for slope in S]
    # print(slope_list)
    
    depth_list = [0 if depth < -16 else
                    3 for depth in z2]
    
    price_factors = [1 + x + y + z for x, y, z in zip(slope_list, depth_list, sediment_list)]
    price_factors = np.round(price_factors,3)
    
    price_m = 800 # €
    
    price_routes = [length * cost_factor * price_m for length, cost_factor in zip(L, price_factors)]
 
    total_price = np.sum(price_routes)
    
    price_INST = total_price/1000000
    
   
    """
    Price of eksploatation
    """
    price_EXP, P = exploatation(L, z2)
    
    print(f'Total installation cost of the route is: {price_INST:.3f} million €')
    print(f'Total operating cost is: {price_EXP:.3f} million €')
    print(f'Total power required for the pump is: {P/1000000:.3f} MW')
    print(f'Number of exceeded constraints: {constraint:.0f}')
    
    """
    Vizualization
    """
    
    if plot:
        # Create a 3D plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = (10, 8), constrained_layout = True)
        
        # Set axis labels
        ax.set_xlabel('Geographic Longitude [°E]')
        ax.set_ylabel('Geographic Latitude [°N]')
        ax.set_zlabel('Sea Depth [m]')
        
        # Plot the surface
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=1, alpha = 0.7, antialiased=True)
        
        # Plot locations with markers and labels
        ax.plot(Ax, Ay, ZA, marker = 'p', ms = 14,color = 'blue', label = 'Dubrovnik')
        ax.plot(Bx, By, ZB, marker = 'p', ms = 14,color = 'black', label = 'Vieste')
        ax.plot(x2, y2, z2, marker = 'o', lw = 1, color = 'purple', alpha = 1, label = 'Pipeline route')
        
        # Plot the green line for the distance between the two cities
        ax.plot([Ax, Bx], [Ay, By], [ZA, ZB], lw = 1, color = 'green', label='Air Distance between Cities')
        
        # Calculate the midpoint between cities and display the distance
        midpoint = ((Ax + Bx) / 2, (Ay + By) / 2, (ZA + ZB) / 2)
        ax.text(midpoint[0], midpoint[1], midpoint[2]+10, f'{haversine(Ax, Ay, Bx, By)/1000:.2f} km', fontsize=10, weight = 'bold')
        
        # Display city names
        ax.text(Ax - 0.075, Ay - 0.075, ZA + 40, 'Dubrovnik', weight='bold')
        ax.text(Bx - 0.1, By - 0.1, ZB - 40, 'Vieste', weight='bold')
        
        # Display pipeline route lenght 
        ax.text(x2[int(len(x2)/2)],y2[int(len(x2)/2)],z2[int(len(x2)/2)]-100, f'{L_SUM/1000:.2f} km', weight = 'bold')
        
        # Set the depth limits
        ax.set_zlim(z_min, z_max)
        ax.set_zlim(z_min, z_max)

        # Find and display the location of the steepest slope
        N_max = np.argmax(S)
        ax.text(x2[N_max] + 0.05, y2[N_max], z2[N_max], f'{S[N_max]:.2f}°', weight='bold')
        ax.plot(x2[N_max], y2[N_max], z2[N_max], marker='x', ms=12, color='red', label='Steepest Slope on Pipeline Segment')
        
        # Display cost and constraint information
        ax.text2D(0.05, 0.95, f'Installation cost = {price_INST:.3f} million €', transform=ax.transAxes)
        ax.text2D(0.05, 0.9, f'Operating cost = {price_EXP:.3f} million €', transform=ax.transAxes)
        ax.text2D(0.05, 0.85, f'Route Length = {L_SUM/1000:.2f} km', transform=ax.transAxes)
        
        if constraint != 0:
            ax.text2D(0.05, 0.8, f'{constraint:.0f} Constraints Violated', transform=ax.transAxes)
        else:
            ax.text2D(0.05, 0.8, 'All Constraints Satisfied', transform=ax.transAxes)
        
        # Set the title and display the legend
        ax.set_title('Optimized Submarine Pipeline Route with Terrain Visualization')
        ax.legend(loc='upper right')
    
        # Uncomment the following lines to set custom view angles and save the plot as an image
        # elevation_angle = 60
        # azimuthal_angle = 20
        # ax.view_init(elevation_angle, azimuthal_angle)
        plt.savefig('Output_Data/Optimized_Pipeline_3D.png', dpi=300)
        plt.show()

    if plot_constraint:
        
        # Open and display the region image
        img = rasterio.open('Input_Data/region.csv')
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        
        image = show(img, ax=ax, cmap='viridis')
        im = image.get_images()[0]
        plt.title('Optimized Submarine Pipeline Route with Bathymetry and Constraints Visualization')
        cbar = plt.colorbar(mappable=im, ax=ax, shrink=0.5, aspect=30)
        cbar.set_label('Sea Depth [m]')
        
        # Plot city locations and labels
        plt.plot(Ax, Ay, ms=12, marker='p', color='red', label='Dubrovnik')
        plt.plot(Bx, By, ms=12, marker='p', color='black', label='Vieste')

        # Set axis labels
        ax.set_xlabel('Geographic Longitude [°E]')
        ax.set_ylabel('Geographic Latitude [°N]')
    
        # Display city names
        ax.text(Ax - 0.18, Ay + 0.1, 'Dubrovnik', weight='bold')
        ax.text(Bx - 0.1, By - 0.1, 'Vieste', weight='bold')

        # Plot the air distance line between cities
        ax.plot([Ax, Bx], [Ay, By], lw=1, color='pink', label='Air Distance between Cities')
        
        # Calculate the midpoint between cities and display the air distance
        midpoint = ((Ax + Bx) / 2, (Ay + By) / 2)
        ax.text(midpoint[0], midpoint[1], f'{haversine(Ax, Ay, Bx, By)/1000:.2f} km', fontsize=10, weight='bold')
        
        # Display pipeline route information
        ax.text(x2[int(len(x2)/2)], y2[int(len(y2)/2)], f'{L_SUM/1000:.2f} km', weight='bold')
        
        # Create custom legend elements for the plot
        custom_lines = [Line2D([0], [0], marker='p', color='w', markerfacecolor='red', markersize=12),
                   Line2D([0], [0], marker='p', color='w', markerfacecolor='k', markersize=12),
                   Patch(facecolor='orange', edgecolor='orange', alpha=1),
                   Patch(facecolor='pink', edgecolor='pink', alpha=1)]

        # Add the legend with custom elements
        fig.legend(custom_lines, ['Dubrovnik', 'Vieste', 'Pipeline Route', 'Air Distance between Cities'], loc='lower right')

        # Plot various constraints from shapefiles with different colors
        shapefiles = ['fishing', 'boreholes', 'corals', 'ship_wrecks', 'submarine_landslides', 'wind_farms']
        colors = ['blue', 'red', 'teal', 'green', 'purple', 'gray']
   
        for i in range(len(shapefiles)):
            shapefile = gpd.read_file(f'Input_Data/constraints/{shapefiles[i]}.shp')
            plot = shapefile.plot(ax=ax, color=colors[i], alpha=1, edgecolor='k', label=shapefiles[i])
    
        # Create a custom legend for constraint types
        shapefile_names = ['Coast and Islands', 'Boreholes', 'Protected Area', 'Shipwrecks', 'Submarine Landslides', 'Wind Farms']
        custom_lines = [Patch(facecolor=colors[0], edgecolor='k', alpha=1),
                        Patch(facecolor=colors[1], edgecolor='k', alpha=1),
                        Patch(facecolor=colors[2], edgecolor='k', alpha=1),
                        Patch(facecolor=colors[3], edgecolor='k', alpha=1),
                        Patch(facecolor=colors[4], edgecolor='k', alpha=1),
                        Patch(facecolor=colors[5], edgecolor='k', alpha=1)]
    
        # Plot the pipeline route
        plt.plot(x2, y2, color='orange', linewidth=2)
    
        # Add the legend for constraint types
        fig.legend(custom_lines, shapefile_names, loc='lower left')
        
        # Uncomment the following line to save the plot as an image
        plt.savefig('Output_Data/Optimized_Pipeline_2D.png', dpi=300)
      
    # penalty = price_INST + price_EXP + constraint

    return price_INST, price_EXP, constraint

def fitness_penalty(design):
    
    P, C = path(design, plot = False)
    
    penalty = P +  C
    
    return penalty
    
if __name__ == '__main__':

    """
    City positions
    """
    Ax, Ay = 18.033, 42.65 # Dubrovnik
    Bx, By = 16.192, 41.89 # Vieste
    
    ZA = bilinear_interpolation([Ax, Ay])[0] # Elevation of point A
    ZB = bilinear_interpolation([Bx, By])[0] # Elevation of point B


    """
    Defining the optimization vector
    """
    n = 200 # number of pipes
    theta_max = np.deg2rad(10)  # Range of angles between pipes
    theta = np.random.uniform(-theta_max, theta_max, n)
    
    """
    Optimization
    """

    import indago
    from indago import PSO
    
    theta0 = theta
    pso = PSO()
    pso.evaluation_function = path
    pso.dimensions = n
    pso.lb = -theta_max
    pso.ub = theta_max
    
    pso.objectives = 2
    pso.constraints = 1
    pso.objective_weights = [1, 1]
    pso.objective_labels = ['Installation Price (MIL €)', 'Operating Price (MIL €)']
    pso.constraint_labels = ['Violated Constraints']
    
    pso.max_iterations = 1000
    pso.max_evaluations = 10000

    pso.monitoring = 'basic'
    pso.X = theta0
    pso.target_fitness = 1e-6
    pso.params['swarm_size'] = 30
    pso.params['inertia'] = 0.8
    pso.params['cognitive_rate'] = 0.7
    pso.params['social_rate'] = 0.8         
    pso.max_stalled_iterations = 200
    
    # pso.convergence_log_file = 'Output_Data/Optimized_Pipeline_pso.path.log'
    
    r = pso.optimize()
    print(r.f)
    REZ = r.X
    np.savetxt('Output_Data/Optimized_Pipeline.txt', REZ, delimiter = ',')
    
    # pso.plot_history()
    # plt.savefig('Output_Data/Optimized_Pipeline_CONV.png', dpi = 300)
    
    # Drawing the optimal path
    path(r.X, plot = True, plot_constraint=True)

    

    """ 
    Loading a specific optimization and visualization
    """
    theta_O = np.loadtxt('Output_Data/Optimized_Pipeline_Best.txt')
    R = path(theta_O, plot = True, plot_constraint= True)
