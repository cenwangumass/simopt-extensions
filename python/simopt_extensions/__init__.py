from .amusement_park import replicate


def patch_model():
    class_name = "simopt.models.amusementpark.AmusementPark"
    function = replicate
    return class_name, function
