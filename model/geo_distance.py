import geopy.distance
import numpy as np

def geodesic_dist(trip):
    pickup_lat = trip['pickup_latitude']
    pickup_long = trip['pickup_longitude']
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    distance = geopy.distance.geodesic((pickup_lat, pickup_long), 
                                       (dropoff_lat, dropoff_long)).miles
    try:
        return distance
    except ValueError:
        return np.nan
    
def circle_dist(trip):
    pickup_lat = trip['pickup_latitude']
    pickup_long = trip['pickup_longitude']
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    distance = geopy.distance.great_circle((pickup_lat, pickup_long), 
                                       (dropoff_lat, dropoff_long)).miles
    try:
        return distance
    except ValueError:
        return np.nan

def jfk_dist(trip):
    jfk_lat = 40.6413
    jfk_long = -73.7781
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    jfk_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (jfk_lat, jfk_long)).miles
    return jfk_distance

def lga_dist(trip):
    lga_lat = 40.7769
    lga_long = -73.8740
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    lga_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (lga_lat, lga_long)).miles
    return lga_distance

def ewr_dist(trip):
    ewr_lat = 40.6895
    ewr_long = -74.1745
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    ewr_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (ewr_lat, ewr_long)).miles
    return ewr_distance

def tsq_dist(trip):
    tsq_lat = 40.7580
    tsq_long = -73.9855
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    tsq_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (tsq_lat, tsq_long)).miles
    return tsq_distance

def cpk_dist(trip):
    cpk_lat = 40.7812
    cpk_long = -73.9665
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    cpk_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (cpk_lat, cpk_long)).miles
    return cpk_distance

def lib_dist(trip):
    lib_lat = 40.6892
    lib_long = -74.0445
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    lib_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (lib_lat, lib_long)).miles
    return lib_distance

def gct_dist(trip):
    gct_lat = 40.7527
    gct_long = -73.9772
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    gct_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (gct_lat, gct_long)).miles
    return gct_distance

def met_dist(trip):
    met_lat = 40.7794
    met_long = -73.9632
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    met_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (met_lat, met_long)).miles
    return met_distance

def wtc_dist(trip):
    wtc_lat = 40.7126
    wtc_long = -74.0099
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    wtc_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (wtc_lat, wtc_long)).miles
    return wtc_distance