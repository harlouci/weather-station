from dataclasses import dataclass


@dataclass
class Experiment:
    """
    A dataclass used to represent an Experiment on MLflow
    Attributes
    ----------
    tracking_server_uri : str
        the URI of MLFlow experiment tracking server
    name : str
        the name of the experiment
    """

    tracking_server_uri: str
    name: str
