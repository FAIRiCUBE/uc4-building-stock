import ast
import math

import folium
import numpy as np
import osmnx as ox
import utm
from typing import List

def utm_bounding_boxes(utmx: np.array, utmy: np.array, radius: float):
    utmllcx = utmx - radius
    utmllcy = utmy - radius
    utmurcx = utmx + radius
    utmurcy = utmy + radius
    return utmllcx, utmllcy, utmurcx, utmurcy

def latlon_bounding_boxes(lats: np.array, lons: np.array, radius: float) -> List[List[float]]:
    # Convert lat lon to utm
    utmx, utmy, utmzone, utmletter = utm.from_latlon(lats, lons)
    # Create utm bounding box
    utmllcx, utmllcy, utmurcx, utmurcy = utm_bounding_boxes(utmx, utmy, radius)
    # Convert the coordinates of the utm bounding box to lat lon
    llclat, llclon = utm.to_latlon(utmllcx, utmllcy, utmzone, utmletter)
    urclat, urclon = utm.to_latlon(utmurcx, utmurcy, utmzone, utmletter)
    # Pack the latlon bounding box coordinates into an list of bounding boxes
    return [[llclat[i], llclon[i], urclat[i], urclon[i]] for i in range(len(llclon))]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #Two methods to define the box
    center = False

    # Center point
    # 1- When Center = True: Define the center and the radius in meters
    if center:
        # Halle
        lat = 51.4954
        lon = 11.9662

        point = (lat, lon)
        # Distance/"radius" in meters
        dist = 1000.0
        south, east, north, west = latlon_bounding_boxes(np.array([lat]), np.array([lon]), dist)[0]
    else:
        # 2- When center = False: define the box manually
        south, east, north, west = 51.3435, 12.5023, 51.6020, 11.4419

    #Get building data from street map.
    buildings = ox.geometries_from_bbox(north, south, east, west, tags={'building': True})

    cols = ['ref:bygningsnr', 'geometry', 'building', 'building:material', 'building:levels', 'capacity',
        'construction', 'height', 'historic', 'roof:height', 'roof:levels',
        'roof:material', 'roof:orientation', 'roof:shape', 'start_date']

    data = buildings#[cols]


    print('BOX: ', "S:", south, ", E:", east, ", N:",north, ", W:",west)
    print("Data Size: ", len(data))
    print("Available Heights: ", len(data[~data['height'].isna()]))
    print("Available Levels: ", len(data[~data['building:levels'].isna()]))
    print("Available Heights & Levels: ", len(data[~data['building:levels'].isna() & ~data['height'].isna()]))
    print("--------------------------------------")

    # Convert building levels from String to Float (problem: it depends on the region/language, e.g., RC means level 0 in French)
    data['building:levels'] = data['building:levels'].str.replace(';', '')
    data['building:levels'] = data['building:levels'].str.replace(',', '')
    data['building:levels'] = data['building:levels'].str.replace('RC', '0')
    data['building:levels'] = data['building:levels'].str.replace('Erdgeschoss und 2. Etage', '2')
    data['building:levels'] = data['building:levels'].str.replace('s', '0')
    data['building:levels'] = data['building:levels'].str.replace('3 - 4', '4').astype(float)

    # Convert height from String to Float
    data['height'] = data['height'].str.replace(',', '.')
    data['height'] = data['height'].str.replace('~', '')
    data['height'] = data['height'].str.replace('m', '').astype(float)
    
    #For some countries ground floor = 0
    data['building:levels'] = data['building:levels'] + 1
    #LevelHeight computing
    data['LevelHeight'] = data['height'] / data['building:levels']

    #First constant: Compute height using the mean as constant
    data['heightComputed'] = data['building:levels'] * data['LevelHeight'].mean()
    data['Distance'] = abs(data['height'] - data['heightComputed'])
    data['Distance*2'] = (data['height'] - data['heightComputed']) ** 2
    subData = data[~data['Distance'].isna()]
    print('------------------------*MEAN-----------------------------')
    error = subData['Distance'].sum() / len(subData)
    error2 = math.sqrt(subData['Distance*2'].sum() / len(subData))
    print('Constant: ', data['LevelHeight'].mean(), 'm')
    print('MAE: ', error, 'm')
    print('RMSE: ', error2, 'm')
    print('Min: ', subData['Distance'].min())
    print('Max: ', subData['Distance'].max())

    #Height = Constant* Building:levels
    constant = 2.4
    Optconstant = data['LevelHeight'].mean()
    OptError = error
    OptError2 = error2

    # Try 20 constants and select the one that yields the best Mean Absolute Error
    for i in range(20):
        print('-------------------------', constant, 'm----------------------------')
        data['heightComputed'] = data['building:levels'] * constant
        data['Distance'] = abs(data['height'] - data['heightComputed'])
        data['Distance*2'] = (data['height'] - data['heightComputed'])**2
        subData = data[~data['Distance'].isna()]

        error = subData['Distance'].sum() / len(subData)
        error2 = math.sqrt(subData['Distance*2'].sum() / len(subData))
        if error < OptError:
            OptError = error
            Optconstant = constant
            OptError2 = error2
        print('Constant: ', constant, 'm')
        print('MAE: ', error, 'm')
        print('RMSE: ', error2, 'm')
        print('Min: ', subData['Distance'].min())
        print('Max: ', subData['Distance'].max())
        constant = constant + 0.1

    print('------------------------OPT-----------------------------')
    print('Constant: ', Optconstant, 'm')
    print('MAE: ', OptError, 'm')
    print('RMSE: ', OptError2, 'm')


    print("MAE: Mean Absolute Error")
    print("RMSE: Root Mean Squared Error")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


