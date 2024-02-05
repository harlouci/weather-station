from pydantic import BaseModel

class Item(BaseModel):
    timestamp:str= ''
    weather_conditions: str= ''
    temperature: float= None
    humidity: float= None
    wind_speed: float= None
    wind_bearing:float= None
    visibility:float= None
    pressure:float= None

json_to_dataframe_col = {'timestamp':'Timestamp',
                         'weather_conditions':'Weather_conditions',
                         'temperature':'Temperature_C',
                         'humidity':'Humidity',
                         'wind_speed':'Wind_speed_kmph',
                         'wind_bearing':'Wind_bearing_degrees',
                         'visibility':'Visibility_km',
                         'pressure':'Pressure_millibars'}

dataframe_col_to_json = {value: key for key, value in json_to_dataframe_col.items()}