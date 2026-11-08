# Neural Network Character Recognition

A Python implementation of character recognition using single and multi-output perceptrons, developed to explore fundamental artificial neural network concepts and learning algorithms.

This project implements and compares three different perceptron architectures for recognizing seven English uppercase letters: **A, B, C, D, E, J, and K**.

## Project Overview

The system represents each character as a binary pixel pattern and uses perceptron learning algorithms to classify the input.

Each character is represented by a **9 × 7 grid**, resulting in 63 input features.

### Pixel Encoding

Each pixel is converted into a numerical value:

| Symbol | Numerical Value |
|---|---:|
| `#` | 1 |
| `.` | -1 |
| Other characters | 0 |

The resulting 63-element vector is used as the input to the neural networks.

## Implemented Architectures

The project explores three different approaches to character classification.

### 1. Single Perceptron (A vs. Not A)

A single perceptron is trained to distinguish the letter **A** from all other characters.

- One output neuron.
- Binary classification.
- Output values: `1`, `-1`, or `0`.
- `1`: The input is classified as A.
- `-1`: The input is classified as Not A.
- `0`: The result is uncertain.

### 2. Multi-Output Perceptron

A single perceptron model with seven output neurons is implemented to recognize all seven characters.

Each output neuron corresponds to one character:

| Output Neuron | Character | Target |
|---|---|---:|
| 1 | A | 1 |
| 2 | B | 2 |
| 3 | C | 3 |
| 4 | D | 4 |
| 5 | E | 5 |
| 6 | J | 6 |
| 7 | K | 7 |

For each training sample, the target vector is constructed using:

- `1` for the correct character.
- `-1` for all other characters.

A character is recognized only when exactly one output neuron produces `1` and all other neurons produce `-1`. Otherwise, the result is classified as unknown.

### 3. Independent Perceptrons

In this approach, seven separate perceptrons are trained independently.

Each perceptron specializes in recognizing one character:

- A vs. Not A
- B vs. Not B
- C vs. Not C
- D vs. Not D
- E vs. Not E
- J vs. Not J
- K vs. Not K

Each network performs binary classification independently.

The final prediction is accepted only when exactly one perceptron recognizes the input and the remaining six reject it.

## Training Configuration

The following parameters are used in the perceptron implementations:

| Parameter | Value |
|---|---:|
| Learning Rate (α) | 1 |
| Threshold (θ) | 0.2 |
| Initial Weights | 0 |
| Initial Bias Weight | 0 |
| Maximum Epochs | 1000 |
| Number of Input Features | 63 |

The weights and bias are updated using the perceptron learning rule whenever the predicted output differs from the target output.

Training stops when the weights no longer change or the maximum number of epochs is reached.

## Project Structure

```text
Neural-Network-Character-Recognition/
│
├── inputs/
│   ├── a1.txt
│   ├── a2.txt
│   ├── ...
│   └── k3.txt
│
├── inputs_multi/
│   ├── a1.txt
│   ├── a2.txt
│   ├── ...
│   └── k3.txt
│
├── perceptron.py
├── MultiOutputPerceptron.py
├── IndependentPerceptrons.py
├── file_handling.py
├── main.py
├── test.py
│
├── inputs.txt
├── inputs_Multi.txt
└── test.txt
```

### Main Files

| File | Description |
|---|---|
| `perceptron.py` | Implements the basic single perceptron and its training algorithm. |
| `MultiOutputPerceptron.py` | Implements the seven-output perceptron. |
| `IndependentPerceptrons.py` | Creates and trains seven independent perceptrons. |
| `file_handling.py` | Handles reading, parsing, and numerical conversion of input files. |
| `main.py` | Initializes and trains the three architectures. |
| `test.py` | Evaluates the trained models using test images. |

## Dataset Format

Each training sample is stored in a separate text file.

The first line contains the target value, followed by nine rows representing the character image.

Example:

```text
1
..##...
...#...
...#...
..#.#..
..#.#..
.#####.
.#...#.
.#...#.
###.###
```

The files `inputs.txt` and `inputs_Multi.txt` contain the paths to the corresponding training samples.

The test dataset is stored in `test.txt`, where individual character images are separated by empty lines.

## Installation and Execution

### Requirements

- Python 3.x
- No external Python libraries are required.

### Run the Project

Clone the repository:

```bash
git clone https://github.com/sonia-heidary/Neural-Network-Character-Recognition.git
```

Navigate to the project directory:

```bash
cd Neural-Network-Character-Recognition
```

Train the three perceptron architectures:

```bash
python main.py
```

Run the testing program:

```bash
python test.py
```

The testing program evaluates each test image using all three approaches and displays their predictions.

## Testing and Comparison

The testing module:

1. Reads character images from `test.txt`.
2. Converts each image into a 63-element input vector.
3. Evaluates the image using the three trained architectures.
4. Displays the classification result of each model.
5. Counts the predictions for each character and unknown inputs.
6. Compares the predictions of the multi-output and independent perceptron models.

The final output includes the agreement count and agreement percentage between the second and third architectures.

## Learning Objectives

This project demonstrates:

- Implementation of the perceptron learning algorithm.
- Weight and bias updates during supervised learning.
- Binary classification using a single perceptron.
- Multi-class classification using multiple output neurons.
- Comparison between a multi-output perceptron and independent binary perceptrons.
- Input preprocessing and feature representation.
- Model evaluation and prediction comparison.

---

**Author:** Sonia Heidary

**Repository:** [Neural Network Character Recognition](https://github.com/sonia-heidary/Neural-Network-Character-Recognition)
