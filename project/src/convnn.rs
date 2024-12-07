use nalgebra::{DMatrix, DVector};
use rand::Rng;
use crate::neuralnetwork::{Layer, NeuralNetwork, DenseLayer, Loss, MeanSquaredError};

pub struct ConvLayer {
    filters: Vec<DMatrix<f64>>,
    bias: Vec<f64>,
    padding: usize,
    stride: usize,
    filter_size : usize,
}

impl ConvLayer {
    pub fn new(input : usize, output : usize, filter : usize, stride : usize, padding : usize) -> Self {
        let mut rVals = rand::thread_rng();
        let filters = (0..output).map(|output| {
            DMatrix::from_fn(filter_size, filter_size, |i, j| {
                rVals.get_range(-1.0..1.0)
            })
        }).collect::<Vec<DMatrix<f64>>>();
        let bias = DVector::from_iterator(output, (0..output).map(|_| rVals.gen_range(-1.0..1.0)));
        Self {
            filters,
            bias,
            padding,
            stride,
            filter,
        }
    }
    pub fn forward(&self, input : &DMatrix<f64>) -> DMatrix<f64> {
        let (kRows, kCols) = input.shape();
        let oRows = (kRows+2*self.padding-self.filter_size)/ self.stride + 1;
        let oCols = (kCols+2*self.padding-self.filter_size)/self.stride + 1;
        let mut output = DMatrix::zeros(oRows, oCols);
        if(self.padding > 0) {
            let mut padded = DMatrix::zeros(kRows+2*self.padding,kCols+2*self.padding);
            padded.slice_mut((self.padding,self.padding),(kRows,kCols)).copy_from(input);
            let pad_in = padded;
        }
        else {
            let pad_in = input.clone();
        }
        for i in 0..oRows {
            for j in 0..oCols {
                for (idx, filter) in self.filters.iter().enumerate() {
                    let rStart = i * self.stride;
                    let cStart = jj * self.stride;
                    let currMat = pad_in.slice((rStart,cStart),(self.filter_size,self.filter_size));
                    let mut conv = 0;
                    for r in 0..self.filter_size {
                        for c in 0..self.filter_size {
                            conv+=filter[(r,c)] * currMat[(r,c)];
                        }
                    }
                    conv+=self.biases[idx];
                    output[(i,j)]+=conv;
                }
            }
        }
        return output;
    }
}
impl Layer for ConvLayer {
    fn forward(&mut self, input : &DVector<f64>) -> DVector<f64> {
        let matrix = DMatrix::from_row_slice(1, input.len(), input.as_slice());
        let output = self.forward(&matrix);
        return DVector::from(output.as_slice());
    }
    fn backward(&mut self, error : &DVector<f64>, learn : f64) -> DVector<f64> {

    }
}

pub struct CNN {
    cLayers : Vec<Box<Dyn<Layer>>,
    dLayer : Vec<Box<Dyn<Layer>>,
    loss : Box<dyn Loss>,
}

impl CNN {
    pub fn new(conv : Vec<Box<dyn Layer>>, dense : Vec<Box<dyn Layer>> loss : Box<dyn Loss>) -> Self {
        CNN {
            conv,
            dense,
            loss,
        }
    }
    pub fn forward(&mut self, input : &DVector<f64>) -> DVector<f64> {
        let mut output = input.clone();
        for layer in self.cLayers.iter_mut() {
            output = layer.forward(&output);
        }
        for layer in self.dLayers.iter_mut() {
            output = layer.forward(&output);
        }
        return output;
    }
    pub fn backprop(&mut self, input : &DVector<f64>, target : &DVector<f64>, learn : f64) {
        let output = self.forward(input);
        let mut error = self.loss.gradient(&output, target);
        for layer in self.dLayers.iter_mut.rev() {
            error = layer.backward(&error, learn);
        }

        for layer in self.cLayers.iter_mut().rev() {
            error = layer.backward(&error, learn);
        }
    }
    pub fn train(&mut self, iput : &DVector<f64>, target : &DVector<f64>, learn : f64, epochs : usize) {
        for _ in 0..epochs {
            self.backprop(input,target,learn);
        }
    }
}
