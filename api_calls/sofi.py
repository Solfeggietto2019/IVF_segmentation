import requests
from typing import List

url = "https://nwndanoruj.execute-api.eu-west-2.amazonaws.com/prod/predict"
validation_key = "peromiraquematangadijolachanga"


def process_json_and_call_sofi(siD_list, aeris_list) -> dict:
    sofi_responses = []

    if len(siD_list) != len(aeris_list):
        raise ValueError(
            "Las listas SiD y Aeris no tienen el mismo número de elementos"
        )
    for sid_item, aeris_item in zip(siD_list, aeris_list):
        if sid_item == None or aeris_item == None:
            print("Got None values for Egg or Sperm")
            continue
        # Extraer los datos necesarios de sid_item
        motility_parameters = sid_item.get("motility_parameters", {})
        morphological_parameters = sid_item.get("morphological_parameters", {})
        # standardize_morph_parameters = sid_item.get("standardize_morph_parameters", {}).get("_StandardMorphologyParameters", {})
        egg_features_parameters = aeris_item.egg_features
        # Crear el diccionario de datos para la llamada al API
        request_data = {
            "body": {
                "VSL": motility_parameters.get("VSL", 0),
                "VCL": motility_parameters.get("VCL", 0),
                "MAD": motility_parameters.get("MAD", 0),
                "LIN": motility_parameters.get("LIN", 0),
                "ALH": motility_parameters.get("ALH", 0),
                "cyto_minor_minoraxis": egg_features_parameters.get(
                    "cyto_minor_minoraxis", 0
                ),
                "cyto_majoraxis": egg_features_parameters.get("cyto_majoraxis", 0),
                "cyto_eccentricity": egg_features_parameters.get(
                    "cyto_eccentricity", 0
                ),
                "cyto_area": egg_features_parameters.get("cyto_area", 0),
                "granu_minor_minoraxis": egg_features_parameters.get(
                    "granu_minor_minoraxis", 0
                ),
                "granu_majoraxis": egg_features_parameters.get("granu_majoraxis", 0),
                "granu_eccentricity": egg_features_parameters.get(
                    "granu_eccentricity", 0
                ),
                "granu_area": egg_features_parameters.get("granu_area", 0),
                "granu_area_rel": egg_features_parameters.get("granu_area_rel", 0),
                "polarbody_minor_minoraxis": egg_features_parameters.get(
                    "polarbody_minor_minoraxis", 0
                ),
                "polarbody_majoraxis": egg_features_parameters.get(
                    "polarbody_majoraxis", 0
                ),
                "polarbody_eccentricity": egg_features_parameters.get(
                    "polarbody_eccentricity", 0
                ),
                "polarbody_area": egg_features_parameters.get("polarbody_area", 0),
                "peri_cyto_minoraxis": egg_features_parameters.get(
                    "peri_cyto_minoraxis", 0
                ),
                "peri_cyto_majoraxis": egg_features_parameters.get(
                    "peri_cyto_majoraxis", 0
                ),
                "peri_cyto_eccentricity": egg_features_parameters.get(
                    "peri_cyto_eccentricity", 0
                ),
                "peri_cyto_area": egg_features_parameters.get("peri_cyto_area", 0),
                "zp_peri_cyto_minoraxis": egg_features_parameters.get(
                    "zp_peri_cyto_minoraxis", 0
                ),
                "zp_peri_cyto_majoraxis": egg_features_parameters.get(
                    "zp_peri_cyto_majoraxis", 0
                ),
                "zp_peri_cyto_eccentricity": egg_features_parameters.get(
                    "zp_peri_cyto_eccentricity", 0
                ),
                "zp_peri_cyto_area": egg_features_parameters.get(
                    "zp_peri_cyto_area", 0
                ),
                "area": morphological_parameters.get("area", 0),
                "perimeter": morphological_parameters.get("perimeter", 0),
                "aspect_ratio": morphological_parameters.get("aspect_ratio", 0),
                "extend": morphological_parameters.get("extend", 0),
                "orientated_angle": morphological_parameters.get("orientated_angle", 0),
                "circularity": morphological_parameters.get("circularity", 0),
                "hull_area": morphological_parameters.get("hull_area", 0),
                "solidity": morphological_parameters.get("solidity", 0),
                "hull_perimeter": morphological_parameters.get("hull_perimeter", 0),
                "convexity": morphological_parameters.get("convexity", 0),
                "eccentricity": morphological_parameters.get("eccentricity", 0),
                "compactness": morphological_parameters.get("compactness", 0),
                "major_axis_radius": morphological_parameters.get(
                    "major_axis_radius", 0
                ),
                "minor_axis_radius": morphological_parameters.get(
                    "minor_axis_radius", 0
                ),
            },
            "validation_key": "peromiraquematangadijolachanga",
            "normalized": False,
        }
        response = requests.post(url, json=request_data)
        sofi_responses.append(response)
    return sofi_responses


def call_sofi(sperms: List, eggs: List) -> list:
    sofi_responses = process_json_and_call_sofi(sperms, eggs)

    return sofi_responses
