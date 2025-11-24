use std::cell::RefCell;
use std::rc::Rc;

use mrg32k3a_rs::Mrg32k3a;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use thiserror::Error;

#[derive(Debug, Error)]
enum Error {
    #[error("invalid inputs: {0}")]
    InvalidInput(String),
}

type Result<T> = std::result::Result<T, Error>;

/// This function is a port of Python's random.gammavariate function.
/// It is necessary to make sure that the result is consistent with Python's
/// implementation for testing purposes.
fn gammavariate(rng: &mut Mrg32k3a, alpha: f64, beta: f64) -> Result<f64> {
    use std::f64::consts::{E, LN_2};
    const LOG4: f64 = 2.0 * LN_2; // log(4)
    const SG_MAGICCONST: f64 = 1.0 + LN_2 / 2.0;

    if alpha <= 0.0 || beta <= 0.0 {
        return Err(Error::InvalidInput(
            "alpha and beta must be > 0.0".to_string(),
        ));
    }

    if alpha > 1.0 {
        // Uses R.C.H. Cheng, "The generation of Gamma
        // variables with non-integral shape parameters",
        // Applied Statistics, (1977), 26, No. 1, p71-74

        let ainv = (2.0 * alpha - 1.0).sqrt();
        let bbb = alpha - LOG4;
        let ccc = alpha + ainv;

        loop {
            let u1 = rng.next_f64();
            if !(1e-7 < u1 && u1 < 0.9999999) {
                continue;
            }
            let u2 = 1.0 - rng.next_f64();
            let v = (u1 / (1.0 - u1)).ln() / ainv;
            let x = alpha * v.exp();
            let z = u1 * u1 * u2;
            let r = bbb + ccc * v - x;
            if r + SG_MAGICCONST - 4.5 * z >= 0.0 || r >= z.ln() {
                return Ok(x * beta);
            }
        }
    } else if alpha == 1.0 {
        // expovariate(1/beta)
        Ok(-(1.0 - rng.next_f64()).ln() * beta)
    } else {
        // alpha is between 0 and 1 (exclusive)
        // Uses ALGORITHM GS of Statistical Computing - Kennedy & Gentle
        loop {
            let u = rng.next_f64();
            let b = (E + alpha) / E;
            let p = b * u;
            let x = if p <= 1.0 {
                p.powf(1.0 / alpha)
            } else {
                -((b - p) / alpha).ln()
            };
            let u1 = rng.next_f64();
            if p > 1.0 {
                if u1 <= x.powf(alpha - 1.0) {
                    break Ok(x * beta);
                }
            } else if u1 <= (-x).exp() {
                break Ok(x * beta);
            }
        }
    }
}

struct ArrivalModel {
    rng: Rc<RefCell<Mrg32k3a>>,
}

impl ArrivalModel {
    fn new(rng: Rc<RefCell<Mrg32k3a>>) -> Self {
        Self { rng }
    }

    fn random(&mut self, lambda: f64) -> Result<f64> {
        if lambda <= 0.0 {
            return Err(Error::InvalidInput("lambda must be > 0.0".to_string()));
        }

        let v = self.rng.borrow_mut().next_f64();
        Ok(-(1.0 - v).ln() / lambda)
    }
}

struct CategoricalModel {
    rng: Rc<RefCell<Mrg32k3a>>,
}

impl CategoricalModel {
    fn new(rng: Rc<RefCell<Mrg32k3a>>) -> Self {
        Self { rng }
    }

    fn random(&mut self, _a: usize, p: &[f64]) -> Result<usize> {
        // TODO: check if p is a valid probability distribution
        let cumsum = p
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect::<Vec<f64>>();

        let v = self.rng.borrow_mut().next_f64();
        for (i, &x) in cumsum.iter().enumerate() {
            if v <= x {
                return Ok(i);
            }
        }

        Ok(cumsum.len() - 1)
    }
}

#[derive(Clone)]
struct ServiceModel {
    rng: Rc<RefCell<Mrg32k3a>>,
}

impl ServiceModel {
    fn new(rng: Rc<RefCell<Mrg32k3a>>) -> Self {
        Self { rng }
    }

    fn random(&mut self, alpha: f64, beta: f64) -> Result<f64> {
        gammavariate(&mut self.rng.borrow_mut(), alpha, beta)
    }
}

#[pyclass]
pub struct Response {
    #[pyo3(get)]
    total_departed: u32,
    #[pyo3(get)]
    percent_departed: f64,
    #[pyo3(get)]
    average_number_in_system: f64,
    #[pyo3(get)]
    attraction_utilization_percentages: Vec<f64>,
}

fn argmin(values: &[f64]) -> (f64, isize) {
    let mut min_value = f64::INFINITY;
    let mut min_idx: isize = -1;
    for (i, &v) in values.iter().enumerate() {
        if v < min_value {
            min_value = v;
            min_idx = i as isize;
        }
    }
    (min_value, min_idx)
}

pub fn replicate(
    factors: &Bound<'_, PyDict>,
    arrival_rng: Rc<RefCell<Mrg32k3a>>,
    attraction_rng: Rc<RefCell<Mrg32k3a>>,
    service_rng: Rc<RefCell<Mrg32k3a>>,
    destination_rng: Rc<RefCell<Mrg32k3a>>,
) -> anyhow::Result<Response> {
    let num_attractions = factors
        .get_item("number_attractions")?
        .unwrap()
        .extract::<usize>()?;
    let arrival_gammas = factors
        .get_item("arrival_gammas")?
        .unwrap()
        .extract::<Vec<f64>>()?;
    let time_open = factors.get_item("time_open")?.unwrap().extract::<f64>()?;
    let erlang_shape = factors
        .get_item("erlang_shape")?
        .unwrap()
        .extract::<Vec<f64>>()?;
    let erlang_scale = factors
        .get_item("erlang_scale")?
        .unwrap()
        .extract::<Vec<f64>>()?;
    // TODO: the upstream code might be wrong
    let queue_capacities: Vec<u32> = factors
        .get_item("queue_capacities")?
        .unwrap()
        .extract::<Vec<f64>>()?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let transition_probabilities = factors
        .get_item("transition_probabilities")?
        .unwrap()
        .extract::<Vec<Vec<f64>>>()?;
    let depart_probabilities = factors
        .get_item("depart_probabilities")?
        .unwrap()
        .extract::<Vec<f64>>()?;

    let mut arrival_model = ArrivalModel::new(arrival_rng);
    let mut attraction_model = CategoricalModel::new(attraction_rng);
    let mut service_models = vec![ServiceModel::new(service_rng.clone()); num_attractions];
    let mut destination_model = CategoricalModel::new(destination_rng);

    // Initialize list of attractions to be selected upon arrival.
    let attraction_range = 0..num_attractions;
    let destination_range = 0..(num_attractions + 1);
    let depart_idx = destination_range.end - 1;

    // Initialize lists of each attraction's next completion time
    let mut completion_times: Vec<f64> = vec![f64::INFINITY; num_attractions];
    let mut min_completion_time: f64 = f64::INFINITY;
    let mut min_completion_idx: isize = -1;

    let mut queues: Vec<u32> = vec![0; num_attractions];

    // Create external arrival probabilities for each attraction.
    let arrival_prob_sum: f64 = arrival_gammas.iter().sum();
    let arrival_probabilities: Vec<f64> = attraction_range
        .clone()
        .map(|i| arrival_gammas[i] / arrival_prob_sum)
        .collect();

    // Initiate clock variables for statistics tracking and event handling.
    let mut clock: f64 = 0.0;
    let mut previous_clock: f64 = 0.0;
    let mut next_arrival: f64 = arrival_model.random(arrival_prob_sum)?;

    // Initialize quantities to track:
    let mut total_visitors: u32 = 0;
    let mut total_departed: u32 = 0;

    // Initialize time average and utilization quantities.
    let mut time_average: f64 = 0.0;
    let mut cumulative_util: Vec<f64> = vec![0.0; num_attractions];

    // Inner helper function to set completion time and update min tracking
    fn set_completion(
        completion_times: &mut [f64],
        min_completion_time: &mut f64,
        min_completion_idx: &mut isize,
        i: usize,
        new_time: f64,
    ) {
        completion_times[i] = new_time;

        if new_time < *min_completion_time {
            *min_completion_time = new_time;
            *min_completion_idx = i as isize;
        } else if (i as isize) == *min_completion_idx {
            // Grab the min index and time with one scanning pass
            let (min_value, min_idx) = argmin(completion_times);
            *min_completion_time = min_value;
            *min_completion_idx = min_idx;
        }
    }

    while clock < time_open {
        // Count number of tourists on attractions and in queues.
        let mut riders: u32 = 0;
        let delta_time: f64 = clock - previous_clock;

        for i in attraction_range.clone() {
            if completion_times[i].is_finite() {
                riders += 1;
                cumulative_util[i] += delta_time;
            }
        }
        let in_system = queues.iter().sum::<u32>() + riders;
        time_average += (in_system as f64) * delta_time;

        previous_clock = clock;

        if next_arrival < min_completion_time {
            // Next event is external tourist arrival.
            total_visitors += 1;

            // Select attraction.
            let attraction_selection: usize =
                attraction_model.random(num_attractions, &arrival_probabilities)?;

            // Check if attraction is currently available.
            if !completion_times[attraction_selection].is_finite() {
                // Generate completion time if attraction available.
                let completion_time = next_arrival
                    + service_models[attraction_selection].random(
                        erlang_shape[attraction_selection],
                        erlang_scale[attraction_selection],
                    )?;
                set_completion(
                    &mut completion_times,
                    &mut min_completion_time,
                    &mut min_completion_idx,
                    attraction_selection,
                    completion_time,
                );
            // If unavailable, check if current queue is less than capacity.
            } else if queues[attraction_selection] < queue_capacities[attraction_selection] {
                queues[attraction_selection] += 1;
            // If queue is full, leave park + 1.
            } else {
                total_departed += 1;
            }
            // Use superposition of Poisson processes to generate next arrival time.
            next_arrival += arrival_model.random(arrival_prob_sum)?;
        } else {
            // Next event is the completion of an attraction.
            // Identify finished attraction (use the tracked min index/time).
            let finished_attraction = min_completion_idx as usize;

            // Pull parameters once (mirrors Python variables alpha/beta)
            let alpha = erlang_shape[finished_attraction];
            let beta = erlang_scale[finished_attraction];

            // Check queue for that attraction.
            if queues[finished_attraction] > 0 {
                let completion_time = min_completion_time
                    + service_models[finished_attraction].random(alpha, beta)?;
                set_completion(
                    &mut completion_times,
                    &mut min_completion_time,
                    &mut min_completion_idx,
                    finished_attraction,
                    completion_time,
                );
                queues[finished_attraction] -= 1;
            } else {
                // If attraction queue is empty, set next completion to infinity.
                set_completion(
                    &mut completion_times,
                    &mut min_completion_time,
                    &mut min_completion_idx,
                    finished_attraction,
                    f64::INFINITY,
                );
            }

            // Check if that person will leave the park.
            // Compose destination probabilities = transition row + depart prob.
            let mut dest_probs = transition_probabilities[finished_attraction].clone();
            dest_probs.push(depart_probabilities[finished_attraction]);

            let next_destination: usize =
                destination_model.random(num_attractions + 1, &dest_probs)?;

            // Check if tourist leaves park.
            if next_destination != depart_idx {
                // If the next attraction is available, start service.
                if !completion_times[next_destination].is_finite() {
                    let completion_time = min_completion_time
                        + service_models[next_destination].random(alpha, beta)?;
                    set_completion(
                        &mut completion_times,
                        &mut min_completion_time,
                        &mut min_completion_idx,
                        next_destination,
                        completion_time,
                    );
                // Else if queue not full, join queue.
                } else if queues[next_destination] < queue_capacities[next_destination] {
                    queues[next_destination] += 1;
                // Else queue full -> leave park.
                } else {
                    total_departed += 1;
                }
            }
        }
        // End of while loop iteration.
        // Check if any attractions are available.
        clock = next_arrival.min(min_completion_time);
    }
    // End of simulation.

    // Calculate overall percent utilization for each attraction.
    for i in attraction_range.clone() {
        cumulative_util[i] /= time_open;
    }

    Ok(Response {
        total_departed,
        percent_departed: (total_departed as f64) / (total_visitors as f64),
        average_number_in_system: time_average / time_open,
        attraction_utilization_percentages: cumulative_util,
    })
}
