use nalgebra::{DMatrix, DVector};
use rand::Rng;

pub trait Loss {
    fn compute(&self, result: &DVector<f64>, test : &DVector<f64>) -> f64;
    fn gradient(&self, result : &DVector<f64>, test : &DVector<f64>) -> DVector<f64>;
}
pub trait Layer {
    fn forward(&mut self, input : &DVector<f64>) -> DVector<f64>; //forward feed
    fn backward(&mut self, error: &DVector<f64>, learn: f64) -> DVector <f64>; //backward feed for backprop
}
pub struct DenseLayer {
    weights : DMatrix<f64>,
    biases : DVector<f64>,
}

impl DenseLayer {
    pub fn new(input : usize, output : usize) -> Self {
        let mut rVals = rand::thread_rng();
        let weights = DMatrix::<f64>::from_iterator(
            output,
            input,
            (0..output*input).map(|_| rVals.gen_range(-0.1..0.1)));
        let biases = DVector::<f64>::from_iterator(
            output,
            (0..output).map(|_| rVals.gen_range(-0.1..0.1)));
        Self {
            weights,
            biases,
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input : &DVector<f64>) -> DVector<f64> {
        let z = &self.weights * input + &self.biases;
        let mut activated = z.clone();
        activated.apply(|x| {*x = x.max(0.0);});
        return activated;
    }
    fn backward(&mut self, error: &DVector<f64>, learn: f64) -> DVector<f64> {
    //   let err_col = DMatrix::from_column_slice(error.len(), 1, error.as_slice());
    //   let gradient_weight = err_col.clone() * DMatrix::from_element(1, self.weights.ncols(), 1.0);
    //   self.weights -= learn * gradient_weight.map(|x| x.clamp(-1e6, 1e6));
    //   self.biases -= learn * error.map(|x| x.clamp(-1e6, 1e6));
    //   return self.weights.transpose() * error();
       
        let err_col = DMatrix::from_column_slice(self.weights.nrows(),1,error.as_slice());
        let gradient_weight = err_col.clone() * DMatrix::from_row_slice(1, self.weights.ncols(), vec![1.0; self.weights.ncols()].as_slice());
        self.weights-= learn * gradient_weight;
        self.biases-=learn*err_col.column(0);
        return DVector::from_element(self.weights.ncols(),0.1);
    }
}
pub struct MeanSquaredError;
impl Loss for MeanSquaredError{
    fn compute(&self, result : &DVector<f64>, test: &DVector<f64>)->f64 {
        let diff = result - test;
        let loss = diff.iter().map(|x| x * x).sum::<f64>()/result.len() as f64;
        if loss.is_nan() {
            eprintln!("NAN DETECTED");
        }
        return loss;
    }
    fn gradient(&self, result : &DVector<f64>, test: &DVector<f64>) -> DVector<f64> {
        let grad = 2.0 * (result-test)/result.len() as f64;
        let grad = grad.map(|x| x.clamp(-1e10, 1e10));
        return grad;
    }
}
impl Clone for MeanSquaredError {
    fn clone(&self) -> Self {
        MeanSquaredError{}
    }
}

pub struct NeuralNetwork {
    layers : Vec<Box<dyn Layer>>,
    loss : Box<dyn Loss>,
}

impl NeuralNetwork {
    pub fn new(layers : Vec<Box<dyn Layer>>, loss : Box<dyn Loss>) -> Self {
        NeuralNetwork {
            layers,
            loss
        }
    }
    pub fn forward(&mut self, input : &DVector<f64>) -> DVector<f64> {
        let mut result = input.clone();
        for layer in self.layers.iter_mut() {
            result = layer.forward(&result);
        }
        return result;
    }
    pub fn backprop(&mut self, input : &DVector<f64>, test : &DVector<f64>, learn : f64) {
        let result = self.forward(input);
        let mut error = self.loss.gradient(&result, test);
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(&error, learn);
        }
    }

    pub fn train(&mut self, input : &DVector<f64>, test : &DVector<f64>, learn : f64, epochs : usize) {
        for _ in 0..epochs {
            self.backprop(input, test, learn);
        }
    }
    pub fn test(&mut self, input : &DVector<f64>, target : &DVector<f64>) -> f64 {
        let result = self.forward(input);
        self.loss.compute(&result, target)
    }
}