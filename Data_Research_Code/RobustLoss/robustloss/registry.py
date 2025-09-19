# registry.py
from .schemas import DatasetSchema, TaskType

REGISTRY: dict[str, DatasetSchema] = {
    "uci_wine": DatasetSchema(
        name="uci_wine",
        target_name="class",
        task_type=TaskType.MULTICLASS,    # 또는 None → 자동 감지
    ),
    "adult": DatasetSchema(
        name="adult",
        target_name="income",
        task_type=TaskType.BINARY,
        categorical_features=("workclass","education","marital-status","occupation",
                              "relationship","race","sex","native-country"),
        drop_features=("fnlwgt",),
    ),
}
