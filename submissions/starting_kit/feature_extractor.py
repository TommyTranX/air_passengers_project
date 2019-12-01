
import pandas as pd
import os
from math import *





class FeatureExtractor(object):
    def __init__(self):
        pass


    def fit(self, X_df, y_array):
        pass


    def transform(self, X_df):
        def haversine(lon1, lat1, lon2, lat2):
            """
            Calculate the great circle distance between two points 
            on the earth (specified in decimal degrees)
            """
            # convert decimal degrees to radians 
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

            # haversine formula 
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a)) 
            r = 6371 # Radius of earth in kilometers. Use 3956 for miles
            return c * r
        X_encoded = X_df.copy()
        path = os.path.dirname(__file__)
        data_weather = pd.read_csv(os.path.join(path, 'external_data.csv'))


        #Weather
        X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC','Mean TemperatureC',
       'Min TemperatureC', 'Dew PointC', 'MeanDew PointC', 'Min DewpointC',
       'Max Humidity', 'Mean Humidity', 'Min Humidity',
       'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa',
       'Min Sea Level PressurehPa', 'Max VisibilityKm', 'Mean VisibilityKm',
       'Min VisibilitykM', 'Max Wind SpeedKm/h', 'Mean Wind SpeedKm/h', 'CloudCover']]
        X_weather = X_weather.rename(
            columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
        
        X_encoded = pd.merge(
            X_encoded, X_weather, how='left',
            left_on=['DateOfDeparture', 'Arrival'],
            right_on=['DateOfDeparture', 'Arrival'],
            sort=False)
        
        #External data but not weather
        X_ext=data_weather[['Date', 'AirPort','Rank_2018','State','city','lat','lng','population','density','ranking','Fuel_price','Holiday']]
        
        Dep_data=X_ext.add_suffix('_Dep')
        Arr_data=X_ext.add_suffix('_Arr')
        
        X_encoded=pd.merge(X_encoded, Dep_data, how='left', left_on=['DateOfDeparture', 'Departure'],
            right_on=['Date_Dep', 'AirPort_Dep'],
            sort=False )
        
        X_encoded=pd.merge(X_encoded, Arr_data, how='left', left_on=['DateOfDeparture', 'Arrival'],
            right_on=['Date_Arr', 'AirPort_Arr'],
            sort=False )
        
        
        #Distance calculation
        X_encoded['Distance']=X_encoded.apply(lambda x:
        haversine(x['lng_Dep'],x['lat_Dep'],x['lng_Arr'],x['lat_Arr']),axis=1)
        
        #Drop useless
        X_encoded=X_encoded.drop([
                                  'AirPort_Dep',
                                   'AirPort_Arr', 
                                  'Fuel_price_Arr','Holiday_Arr',
                                'lat_Dep','lng_Dep','lat_Arr','lng_Arr','Date_Dep','Date_Arr'], axis=1)


        X_array = X_encoded.values
        return X_encoded    
