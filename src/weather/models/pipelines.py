from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from weather.features.dataframe_transformer import SimpleCustomPipeline


@dataclass
class FullPipeline:

    cleaning_pipeline: Pipeline
    target_pipelie: Pipeline
    input_pipeline: SimpleCustomPipeline
