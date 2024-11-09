use nalgebra::{DMatrix, DVector};
pub trait Loss {
    fn compute(&self, result: &DVector<f64>, test : &DVector<f64>) -> f64;
    fn gradient(&self, result : &DVector<f64>, test : &DVector<f64>) -> DVector<f64>;
}
pub trait Layer {
    fn forward(&self, input : &DVector<f64>) -> DVector<f64> //forward feed
    fn backward(&self, error: &DVector<f64>, learn: f64) -> DVector <f64> //backward feed for backprop

}
pub struct DenseLayer {
    weights : DMatrix<f64>,
    biases : DVector<f64>,
}

impl DenseLayer {
    pub fn new(input : usize, output : usize) -> Self {
        let weights = DMatrix::<f64>::new_random(output, input);
        let biases = DVector::<f64>::new_random(output);
    }
}

impl Layer for DenseLayer {
    fn forward(&self, input : DVector<f64>) -> DVector<f64> {
        //IMPLEMENT FEED FORWARD
        //TODO
    }
    fn backward(&self, error: &DVector<f64>, learn: <f64>) -> DVector<f64> {
        //IMPLEMENT BACK PROPOGATION
        //TODO
    }
}

pub struct MeanSquaredError;
impl LossFunction for MeanSquaredError {
 fn compute(&self, result : &DVector<f64>, test : &DVector<f64>) -> f64 {
    //TODO
 }   
 fn gradient(&self, result : &DVector<f64>, test : &DVector<f64>) -> DVector<f64> {
    //TODO
 }
}

// More Loss Functions TBA

pub struct NeuralNetwork {
    layers : Vec<Box<dyn Layer>>,
    loss : Box<dyn LossFunction>,
}

impl NeuralNetwork {
    pub fn new(layers : Vec<Box<dyn Layer>>, loss : Box<dyn LossFunction>) -> Self {
        Neural Network {
            layers,
            loss
        }
    }
    pub fn forward(&self, input : DVector<f64>) -> DVector<f64> {
        let mut result = input.clone();
        for layer in &self.layers {
            result = layer.forward(&result);
        }
        return result;
    }
    pub fkn backprop(&mut self, input : &DVector<f64>, test : &DVector<f64>, learn : f64) {
        //TODO
    }
    pub fn train(&mut self input : &DVector<f64>, test : &DVector<f64>, learn : f64, epochs : usize) {
        for i in 0..epochs {
            self.backprop(input, target, learn);
        }
    }
}