import numpy as np
import json


with open("data/json/morphology_features_stats.json", "r") as file:
    motility_feature_stats = json.load(file)


def standardize_sperm_metrics(
    sperm_metrics, motility_feature_stats=motility_feature_stats
):
    key = "('20X', '[640.0, 480.0]', '7%')"  # Settings Key
    if key not in motility_feature_stats:
        raise ValueError(
            f"La clave {key} no se encuentra en el archivo de estandarización."
        )

    standardized_metrics = {}
    stats_for_key = motility_feature_stats[key]

    for feature, value in sperm_metrics.items():
        feature_key = feature + "_mean"
        if feature_key in stats_for_key:
            stats = stats_for_key[feature_key]
            lambda_val = stats["lambda"]
            mean = stats["mean"]
            scale = stats["scale"]

            # Yeo-Johnson transform
            transformed_value = yeo_johnson_transform(np.array([value]), lambda_val)[0]

            # Estandarizar el valor
            standardized_value = standardize(transformed_value, mean, scale)

            standardized_metrics[feature] = standardized_value
        else:
            standardized_metrics[feature] = (
                value  # Si no hay estandarizacion, dejar el valor original
            )
    return {"_StandardMorphologyParameters": standardized_metrics}


def yeo_johnson_transform(x, lmbda):
    pos = x >= 0
    if lmbda != 2:
        x[pos] = (
            ((x[pos] + 1) ** lmbda - 1) / lmbda if lmbda != 0 else np.log(x[pos] + 1)
        )
    if lmbda != 0:
        x[~pos] = (
            -((-x[~pos] + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
            if lmbda != 2
            else -np.log(-x[~pos] + 1)
        )
    return x


def standardize(x, mean, scale):
    return (x - mean) / scale
