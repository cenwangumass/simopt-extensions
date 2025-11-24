mod amusement_park;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod simopt_extensions {
    use pyo3::prelude::*;

    #[pymodule]
    mod _amusement_park {
        use std::cell::RefCell;
        use std::rc::Rc;

        use pyo3::prelude::*;
        use pyo3::types::PyDict;

        use crate::amusement_park;

        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn replicate(
            factors: &Bound<'_, PyDict>,
            arrival_seed: Vec<u32>,
            arrival_indices: Vec<usize>,
            // TODO: attraction seed and indices are shared with arrival. This
            // might be a bug in the upstream code.
            _attraction_seed: Vec<u32>,
            _attraction_indices: Vec<usize>,
            service_seed: Vec<u32>,
            service_indices: Vec<usize>,
            destination_seed: Vec<u32>,
            destination_indices: Vec<usize>,
        ) -> anyhow::Result<amusement_park::Response> {
            let arrival_rng = Rc::new(RefCell::new(mrg32k3a_rs::Mrg32k3a::new(
                &arrival_seed.try_into().unwrap(),
                &arrival_indices.try_into().unwrap(),
            )));
            let attraction_rng = arrival_rng.clone();
            let service_rng = Rc::new(RefCell::new(mrg32k3a_rs::Mrg32k3a::new(
                &service_seed.try_into().unwrap(),
                &service_indices.try_into().unwrap(),
            )));
            let destination_rng = Rc::new(RefCell::new(mrg32k3a_rs::Mrg32k3a::new(
                &destination_seed.try_into().unwrap(),
                &destination_indices.try_into().unwrap(),
            )));
            amusement_park::replicate(
                factors,
                arrival_rng,
                attraction_rng,
                service_rng,
                destination_rng,
            )
        }
    }
}
