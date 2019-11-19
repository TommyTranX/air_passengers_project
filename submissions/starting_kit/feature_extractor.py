
import pandas as pd
import os


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df.copy()
        path = os.path.dirname(__file__)
        data_weather = pd.read_csv(os.path.join(path, 'external_data.csv'))
        X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC']]
        X_weather = X_weather.rename(
            columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
        
        X_encoded = pd.merge(
            X_encoded, X_weather, how='left',
            left_on=['DateOfDeparture', 'Arrival'],
            right_on=['DateOfDeparture', 'Arrival'],
            sort=False)
        
        #External data but not weather
        X_ext=data_weather[['Date', 'AirPort','Rank_2018','State','city','lat','lng','population','density','ranking']]
        Dep_data=X_ext.add_suffix('_Dep')
        Arr_data=X_ext.add_suffix('_Arr')
        
        X_encoded=pd.merge(X_encoded, Dep_data, how='left', left_on=['DateOfDeparture', 'Arrival'],
            right_on=['Date_Dep', 'AirPort_Dep'],
            sort=False )
        X_encoded=pd.merge(X_encoded, Arr_data, how='left', left_on=['DateOfDeparture', 'Arrival'],
            right_on=['Date_Arr', 'AirPort_Arr'],
            sort=False )
        
        #Dummy Depart/Arr
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(
            pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        
        #External
        
        
       # X_encoded=X_encoded.join(Dep_data, on='Departure', how='left')
       # X_encoded=X_encoded.join(Arr_data, on='Arrival', how='left')
        
        
        '''#Dummy external
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['State_Dep'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['State_Arr'], prefix='a'))
        
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['Major city served_Dep'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['Major city served_Arr'], prefix='a'))
        X_encoded=X_encoded.drop(['Airports (large hubs)_Dep', 'Airports (large hubs)_Arr','Major city served_Arr','Major city served_Dep','State_Arr','State_Dep'], axis=1)
        '''
        
        
        #Dummy Date
        X_encoded['Weekend'] = ((pd.DatetimeIndex(X_encoded['DateOfDeparture']).dayofweek) // 5 == 1).astype(float)
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        #We one hot encode all those
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        
        X_encoded=X_encoded.drop(['Date_Dep','Date_Arr','city_Dep','city_Arr','AirPort_Dep','State_Dep', 'AirPort_Arr', 'State_Arr'], axis=1)
        
        
        X_array = X_encoded.values
        return X_array
