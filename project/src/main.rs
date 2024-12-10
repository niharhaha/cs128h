use nalgebra::{DMatrix, DVector};
use rand::Rng;
mod neuralnetwork;
use neuralnetwork::{NeuralNetwork, DenseLayer, Loss, MeanSquaredError};
pub fn generate_data(samples : usize, features : usize, classes : usize) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
    let mut rVals = rand::thread_rng();
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for _ in 0..samples {
        let image = DVector::from_iterator(features, (0..features).map(|_| rVals.gen_range(-1.0..1.0)));
        let label = DVector::from_fn(classes, |i, _| {
            if i==rVals.gen_range(0..classes) {
                1.0
            }
            else {0.0}
        });
        images.push(image);
        labels.push(label);
    }
    return (images, labels);
}

fn main() {
    let num_samples = 100;
    let num_features = 784;
    let num_classes = 10; 

    let (train_images, train_labels) = generate_data(num_samples, num_features, num_classes);
    let (test_images, test_labels) = generate_data(num_samples / 2, num_features, num_classes);

    let layer_1 = DenseLayer::new(num_features, 128); 
    let layer_2 = DenseLayer::new(128, num_classes);

    let mut NeuralNet = NeuralNetwork::new(
        vec![
            Box::new(layer_1),
            Box::new(layer_2),
        ],
        Box::new(MeanSquaredError),
    );

    let epochs = 10;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for (i, image) in train_images.iter().enumerate() {
            let label = &train_labels[i];
            NeuralNet.train(image, label, 0.001, 1);
            let loss = NeuralNet.test(image, label);
            total_loss+=loss;
        }
        let avg_loss = total_loss/train_images.len() as f64;
        println!("Epoch {} : Average Training Loss: {} ", epoch+1, avg_loss);
    }
    let mut total_loss = 0.0;
    for (i, image) in test_images.iter().enumerate() {
        let label = &test_labels[i];
        let loss = NeuralNet.test(image, label);
        total_loss += loss;
       // println!("Test {}: Loss: {}", i+1, loss);
    }
    let avg_loss = total_loss / test_images.len() as f64;
    println!("Average Loss: {}", avg_loss);
}
