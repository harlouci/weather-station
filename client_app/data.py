from pydantic import BaseModel

class Item(BaseModel):
    S_No:int
    Timestamp:str= ''
    Location:str
    Temperature_C: float = None
    Apparent_Temperature_C: float = None
    Humidity: float = None
    Wind_speed_kmph: float = None
    Wind_bearing_degrees: float = None
    Visibility_km: float = None
    Pressure_millibars: float = None
    Weather_conditions: str = ''


"""

class Item(BaseModel):
    timestamp:str= ''
    weather: str= ''
    temperature: float= None
    humidity: float= None
    wind_speed: float= None
    wind_bearing:float= None
    visibility:float= None
    pressure:float= None

json_to_dataframe_col = {'timestamp':'Timestamp',
                         'weathers':'Weather_conditions',
                         'temperature':'Temperature_C',
                         'humidity':'Humidity',
                         'wind_speed':'Wind_speed_kmph',
                         'wind_bearing':'Wind_bearing_degrees',
                         'visibility':'Visibility_km',
                         'pressure':'Pressure_millibars'}

dataframe_col_to_json = {value: key for key, value in json_to_dataframe_col.items()}
"""
