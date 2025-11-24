from simopt_extensions.simopt_extensions import _amusement_park


def replicate(self):
    arrival_seed = self.arrival_model.rng.ref_seed
    arrival_indices = self.arrival_model.rng.s_ss_sss_index
    attraction_seed = self.attraction_model.rng.ref_seed
    attraction_indices = self.attraction_model.rng.s_ss_sss_index
    service_seed = self.service_models[0].rng.ref_seed
    service_indices = self.service_models[0].rng.s_ss_sss_index
    destination_seed = self.destination_model.rng.ref_seed
    destination_indices = self.destination_model.rng.s_ss_sss_index

    response = _amusement_park.replicate(
        self.factors,
        arrival_seed,
        arrival_indices,
        attraction_seed,
        attraction_indices,
        service_seed,
        service_indices,
        destination_seed,
        destination_indices,
    )

    return {
        "total_departed": response.total_departed,
        "percent_departed": response.percent_departed,
        "average_number_in_system": response.average_number_in_system,
        "attraction_utilization_percentages": response.attraction_utilization_percentages,
    }, {}
