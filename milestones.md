# Milestones

## 25.07
### Overfitting Results
  - Trained WGAN-GP (with Auxillary Classifier) on the following images:
    - <img src="./food-samples/overfit/1001116.jpg" width="64px"/> <img src="./food-samples/overfit/1008104.jpg" width="64px"/> <img src="./food-samples/overfit/1008144.jpg" width="64px"/> <img src="./food-samples/overfit/1006982.jpg" width="64px"/> <img src="./food-samples/overfit/1008491.jpg" width="64px"/> <img src="./food-samples/overfit/1009131.jpg" width="64px"/>
  - Results:
    - <img src="./results/190621_1614_pizza_pancakes_small/samples_9999.png"/>
  - Metrics:
    - <img src="./results/190621_1614_pizza_pancakes_small/train_disc_cost.jpg"/>
    - <img src="./results/190621_1614_pizza_pancakes_small/train_gen_cost.jpg"/>
    - <img src="./results/190621_1614_pizza_pancakes_small/wasserstein_distance.jpg"/>

### First run with all samples from Pizza and Pancakes
- Generated images
  - <img src="./results/190621_1832_pizza_pancakes_big/samples_9999.png"/>
- Metrics:
  - <img src="./results/190621_1832_pizza_pancakes_big/train_disc_cost.jpg"/>
  - <img src="./results/190621_1832_pizza_pancakes_big/train_gen_cost.jpg"/>
  - <img src="./results/190621_1832_pizza_pancakes_big/wasserstein_distance.jpg"/>

### Dataset Switch
- The image quality of samples from Food 101 might not be good enough:
  - only 1k samples per class
  - bad lightening
  - different angles
  - differen scales
  - includes many other (non-food) objects 
  - Comparison:
    - Burger data: Google (left) vs Food 101 (right)
<img src="./food-samples/dataset-switch/burger-google.png" width="350px"/><img src="./food-samples/dataset-switch/burger-food-101.png" width="350px"/>
    - Pizza data: Google (left) vs Food 101 (right)
<img src="./food-samples/dataset-switch/pizza-google.png" width="350px"/><img src="./food-samples/dataset-switch/pizza-food-101.png" width="350px"/>
- => Create our own dataset from google
  - use [Google Images Download](https://github.com/hardikvasa/google-images-download)
  - Start with two classes (pizza, burger)
  - download > 10k images per class


## 18.06
- **Basic Model Implementation**
  - We have chosen *PyTorch* as the deep learning framework since its dynamic computation graph and eager evaluation allows us to quickly protoype and debug our experiments
  - As base implementation, we will use a [PyTorch implementation](https://github.com/jalola/improved-wgan-pytorch) of WGAN-GP

- **Next Goals**
  - Train first model on Food-101 dataset
  - Save training weights
  - Visualize generated images
  - Come up with an idea for conditional interpolation between classes

## 11.06 
- **Main goal definition**: 
  - Generate conditioned images of food  
  - Interpolate between different classes of food using the labels as condition

- **Model**:
  - We wil use WGAN-GP proposed in [Improved Training of Wasserstein GAN](https://arxiv.org/abs/1704.00028)

- **Dataset**:
  - We have chosen the Food 101 dataset by [Bossard et al. ](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
  - Contains 101 different classes of food, each represented by 1000 images
  - *Samples*
    - <img src="./food-samples/1005649.jpg" width="50px"/> <img src="./food-samples/100076.jpg" width="50px"/> <img src="./food-samples/100057.jpg" width="50px"/> <img src="./food-samples/1003501.jpg" width="50px"/> <img src="./food-samples/1006121.jpg" width="50px"/> 

- **Next Goals**
  - Decide for Deep Learning framework
  - Find a robust base implementation of [Improved Training of Wasserstein GAN](https://arxiv.org/abs/1704.00028) in that framework

