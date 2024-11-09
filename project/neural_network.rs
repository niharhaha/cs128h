use nalgebra::{DMatrix, DVector};

pub stuct Network {
    input: usize, //Num Inputs
    hidden: usize, //Num Hidden
    output: usize, //Num Outputs
    input_weight: DMatrix<f64>, //Weight of Each Input
    output_weight: DMatrix<f64>, //Weight of Each Output
    bias: DVector<f64>,
    output_bias: DVector<f64>,
}

impl Network {
    pub fn new(input, hidden, output) -> Self {
        let input_weight = DMatrix::<f64>::new_random(hidden, input);
        let output_weight = DMatrix::<f64>::new_random(output, hidden);
        let bias = DVector::<f64>::new_random(hidden);
        let output_bias = DVector::<f64>::new_random(output);
        Network {
            input,
            hidden,
            output,
            input_weight,
            output_weight,
            bias,
            output_bias,
        }
    }
}