## Tic Tac Toe

This is a Dueling Double Deep Q Network (DDDQN) that learns to play Tic-Tac-Toe perfectly after completing training. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

```
pip install numpy
pip install matplotlib
pip install tensorflow
pip install keras
```

### Installing

Clone the repository

```
git clone https://github.com/XanderBys/Tic_Tac_Toe.git
```

To train the model, run

```
python3 train.py NUM_ROUNDS*BATCH_SIZE p1_dueling p2_dueling p1_PER p2_PER
```

About 1 000 000 rounds of training with batch size of 100 is needed for optimal performance.

## Running the tests

Run run_tests.py and enter the input data after the model had been trained.

## Built With

* [Tensorflow](https://www.tensorflow.org/api_docs) - Backend for Keras
* [Keras](https://keras.io/api/) - Framework to create neural networks
* [NumPy](https://numpy.org/doc/) - Used for matrix operations

## Authors

* **Xander Bystrom** - [XanderBys](https://github.com/XanderBys)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
