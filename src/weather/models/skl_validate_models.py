import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from deepchecks.tabular import Dataset
from deepchecks.tabular import Suite
from deepchecks.tabular.checks import TrainTestPerformance
from deepchecks.tabular.checks import SimpleModelComparison


def min_perf_validation(dataset, data_transformer, classifier, cat_features):
    ds_train = Dataset(dataset.train_x, label=dataset.train_y, cat_features=cat_features)
    ds_test =  Dataset(dataset.val_x,  label=dataset.val_y, cat_features=cat_features)
    pipeline = Pipeline([('transformer', data_transformer), ('classifier', classifier)])
    custom_suite = Suite('Pipeline Test Suite',
                            # add examples of simple checks
                            TrainTestPerformance()\
                            .add_condition_train_test_relative_degradation_less_than(
                                threshold=0.15
                            )\
                            .add_condition_test_performance_greater_than(0.8),
                            SimpleModelComparison(
                                strategy='most_frequent'
                                ).add_condition_gain_greater_than(0.3)
                        )
    result = custom_suite.run(model=pipeline, 
                                train_dataset=ds_train, 
                                test_dataset=ds_test)
    return result
