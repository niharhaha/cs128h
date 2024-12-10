use nalgebra::{DMatrix, DVector};
use rand::Rng;
use crate::neuralnetwork::{Layer, NeuralNetwork, DenseLayer, Loss, MeanSquaredError};

pub struct ConvLayer {
    filters: Vec<DMatrix<f64>>,
    bias: DVector<f64>,
    padding: usize,
    stride: usize,
    filter_size : usize,
}

impl ConvLayer {
    pub fn new(input : usize, output : usize, filter_size : usize, stride : usize, padding : usize) -> Self {
        let mut rVals = rand::thread_rng();
        let filters = (0..output).map(|output| {
            DMatrix::from_fn(filter_size, filter_size, |i, j| {
                rVals.gen_range(-1.0..1.0)
            })
        }).collect::<Vec<DMatrix<f64>>>();
        let bias = DVector::from_iterator(output, (0..output).map(|_| rVals.gen_range(-1.0..1.0)));
        Self {
            filters,
            bias,
            padding,
            stride,
            filter_size,
        }
    }
    pub fn forward(&self, input : &DMatrix<f64>) -> DMatrix<f64> {
        let (kRows, kCols) = input.shape();
        let oRows = (kRows+2*self.padding-self.filter_size)/ self.stride + 1;
        let oCols = (kCols+2*self.padding-self.filter_size)/self.stride + 1;
        let mut output = DMatrix::zeros(oRows, oCols);
        let pad_in = match self.padding {
            0 => input.clone(),
            _ => {
                let mut padded = DMatrix::zeros(kRows+2*self.padding,kCols+2*self.padding);
                padded.slice_mut((self.padding,self.padding),(kRows,kCols)).copy_from(input);
                padded
            }
        };
        for i in 0..oRows {
            for j in 0..oCols {
                for (idx, filter) in self.filters.iter().enumerate() {
                    let rStart = i * self.stride;
                    let cStart = j * self.stride;
                    let currMat = pad_in.slice((rStart,cStart),(self.filter_size,self.filter_size));
                    let mut conv : f64 = 0 as f64;
                    for r in 0..self.filter_size {
                        for c in 0..self.filter_size {
                            conv+=filter[(r,c)] * currMat[(r,c)];
                        }
                    }
                    conv+=self.bias[idx];
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
        let output = self.forward(&DVector::from_column_slice(&matrix.as_slice()));
        return DVector::from_iterator(output.len(), output.iter().cloned());
    }
    fn backward(&mut self, error : &DVector<f64>, learn : f64) -> DVector<f64> {
        let err_mat = DMatrix::from_row_slice(1, error.len(), error.as_slice());
        let (iRows, iCols) = err_mat.shape();
        let oSize = self.filters.len();

        let mut iGradient = DMatrix::zeros(iRows, iCols);
        let mut fGradient = vec![DMatrix::zeros(self.filter_size, self.filter_size); oSize];
        let mut bGradient : DVector<f64> = DVector::zeros(oSize);

        let pad_in = match self.padding {
            0 => err_mat.clone(),
            _=>{
                let mut padded = DMatrix::zeros(iRows+2*self.padding, iCols + 2*self.padding,);
                padded.slice_mut((self.padding,self.padding),(iRows, iCols)).copy_from(&err_mat);
                padded
            }
        };

        for idx in 0..oSize {
            for i in 0..iRows {
                for j in 0..iCols {
                    for fRow in 0..self.filter_size {
                        for fCol in 0..self.filter_size {
                            let row = i * self.stride + fRow;
                            let col = j * self.stride + fCol;
                            if(row < pad_in.nrows() && col < pad_in.ncols()) {
                                fGradient[idx][(fRow, fCol)] +=pad_in[(row,col)] * err_mat[(i,j)];
                            }
                        }
                    }
                    bGradient[idx] +=err_mat[(i,j)];
                    for fRow in 0..self.filter_size {
                        for fCol in 0..self.filter_size {
                            let row = i * self.stride + fRow;
                            let col = j * self.stride + fCol;

                            if(row < iGradient.nrows() && col < iGradient.ncols()) {
                                iGradient[(row,col)] +=self.filters[idx][(fRow,fCol)] * err_mat[(i,j)];
                            }
                        }
                    }
                }
            }
        }
        for idx in 0..oSize {
            self.filters[idx] -= learn * fGradient[idx].clone();
            self.bias[idx] -= learn * bGradient[idx];
        }
        return DVector::from_iterator(iGradient.len(), iGradient.iter().cloned());
        }
    }
pub struct CNN {
    cLayers : Vec<Box<dyn Layer>>,
    dLayers : Vec<Box<dyn Layer>>,
    loss : Box<dyn Loss>,
}

impl CNN {
    pub fn new(cLayers : Vec<Box<dyn Layer>>, dLayers : Vec<Box<dyn Layer>>, loss : Box<dyn Loss>) -> Self {
        CNN {
            cLayers,
            dLayers,
            loss,
        }
    }
    pub fn getLoss(&self) -> &dyn Loss {
        return &*self.loss;
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
        for layer in self.dLayers.iter_mut().rev() {
            error = layer.backward(&error, learn);
        }

        for layer in self.cLayers.iter_mut().rev() {
            error = layer.backward(&error, learn);
        }
    }
    pub fn train(&mut self, input : &DVector<f64>, target : &DVector<f64>, learn : f64, epochs : usize) {
        for _ in 0..epochs {
            self.backprop(input,target,learn);
        }
    }
}
