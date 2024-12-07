use nalgebra::{DMatrix, DVector};
use rand::Rng;

pub trait Loss {
    fn compute(&mut self, result: &DVector<f64>, test : &DVector<f64>) -> f64;
    fn gradient(&mut self, result : &DVector<f64>, test : &DVector<f64>) -> DVector<f64>;
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
            (0..output*input).map(|_| rVals.gen_range(-1.0..1.0)));
        let biases = DVector::<f64>::from_iterator(
            output,
            (0..output).map(|_| rVals.gen_range(-1.0..1.0)));
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
        for i in 0..activated.len() {
            activated[i]=activated[i].max(0.0);
        }
        return activated;
    }
    fn backward(&mut self, error: &DVector<f64>, learn: f64) -> DVector<f64> {
        let delta = error.component_mul(&self.weights);
        let gradient_weight = delta.clone() * error.transpose();
        self.weights-=learn*gradient_weight;
        self.biases-=learn*delta.clone();
        return self.weights.transpose()*delta.clone();
    }
}
pub struct MeanSquaredError;
impl Loss for MeanSquaredError{
    fn compute(&mut self, result : &DVector<f64>, test: &DVector<f64>)->f64 {
        let diff = result - test;
        return diff.iter().map(|x| x * x).sum::<f64>()/result.len() as f64;
    }
    fn gradient(&mut self, result : &DVector<f64>, test: &DVector<f64>) -> DVector<f64> {
        return 2.0 * (result-test)/result.len() as f64;
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
}