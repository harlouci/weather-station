from pydantic import BaseModel


class Item(BaseModel):
    S_No:int
    Timestamp:str= ""
    Location:str
    Temperature_C: float = None
    Apparent_Temperature_C: float = None
    Humidity: float = None
    Wind_speed_kmph: float = None
    Wind_bearing_degrees: float = None
    Visibility_km: float = None
    Pressure_millibars: float = None
    Weather_conditions: str = ""

