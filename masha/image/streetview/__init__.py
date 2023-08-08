from .streeview import StreetView
from .panorama import Panorama
from masha.config import app_config


__all__ = ["StreetView", "Panorama"]


StreetView.register(app_config.google.maps_api_key, app_config.google.maps_api_secret)
Panorama.register(app_config.google.maps_api_key)