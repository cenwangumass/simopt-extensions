from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.models.amusementpark import AmusementPark
from simopt_extensions.amusement_park import replicate


def test_amusement_park():
    model = AmusementPark()
    rng_list = [MRG32k3a(s_ss_sss_index=[0, i, 0]) for i in range(model.n_rngs)]
    model.before_replicate(rng_list)
    responses_py, gradients_py = model.replicate()
    responses_rust, gradients_rust = replicate(model)
    assert responses_py == responses_rust
    assert gradients_py == gradients_rust
