[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Realtime Hand Gesture Recognition️

This project presents a from-scratch implementation of a real-time hand gesture recognition system using neural networks and spatial pose data. The system recognizes gestures like *swipe left*, *swipe right*, *clockwise rotation*, and more from video input. It enables gesture-based control of a presentation slideshow interface in real time.

The gesture detection model is served by a lightweight [Sanic](https://sanic.dev/) WebSocket server, which streams predictions to the browser. The presentation itself is rendered using [Reveal.js](https://revealjs.com/), a modern HTML-based presentation framework. Recognized gestures are mapped to Reveal.js API methods to allow fluid, hands-free slide navigation and interaction.

&#x20;

## Table of Contents

* [Project Overview](#project-overview)
* [Getting Started](#getting-started)
* [Install Requirements](#install-requirements)
* [Performance & Metrics](#performance--metrics)
* [Confusion Matrices](#confusion-matrices-1)
* [Project Structure](#project-structure)
* [Dataset Details](#dataset-details)
* [License](#license)
---

## Project Overview

The system is composed of several components and technologies. It combines a Sanic-based WebSocket server for real-time gesture communication with a Reveal.js-powered browser presentation. The Sanic server handles gesture prediction and streams events to the frontend, which updates the presentation accordingly.

- **Dataset Creation**: Custom-recorded videos with annotated gesture intervals using ELAN.
- **Pose Extraction**: Using [MediaPipe](https://google.github.io/mediapipe/) to extract 3D keypoints of the upper body.
- **Feature Engineering**: Temporal features computed from keypoints.
- **Model Training**: A neural network built with NumPy to classify sequences into gesture categories.
- **Application Interface**: Gesture-based control of a slideshow application.

---

## Getting Started

### Install Requirements

```
pip install -r requirements.txt
```

### 1. **Basic Gesture Control Usage**


Launch the Sanic server, open the Reveal.js slideshow, and begin webcam input:

```bash
python slideshow/slideshow_demo.py
```

#### Supported gestures:
| **Gesture Name**     | **Reveal.js Method / Action**                            | **Description**                                                        |
|----------------------| -------------------------------------------------------- | ---------------------------------------------------------------------- |
| `swipe_right`        | `Reveal.right()`                                         | Go to the next horizontal slide                                        |
| `swipe_left`         | `Reveal.left()`                                          | Go to the previous horizontal slide                                    |
| `rotate (clockwise)` | `Reveal.slide(0, 0)`<br>`rotateRotatables(currentSlide)` | Jump to the first slide and trigger custom rotation logic for elements |


### 2. **Extended Gesture Control Usage**


Include additional gestures with:

```bash
python slideshow/slideshow_demo_optionals.py
```
#### Supported gestures:

| **Gesture Name**     | **Reveal.js Method / Action**                            | **Description**                                                        |
|----------------------| -------------------------------------------------------- | ---------------------------------------------------------------------- |
| `swipe_right`        | `Reveal.right()`                                         | Go to the next horizontal slide                                        |
| `swipe_left`         | `Reveal.left()`                                          | Go to the previous horizontal slide                                    |
| `rotate (clockwise)` | `Reveal.slide(0, 0)`<br>`rotateRotatables(currentSlide)` | Jump to the first slide and trigger custom rotation logic for elements |
| `pinch`              | `Reveal.toggleOverview()`                                | Toggle overview mode (zoom out or back in)                             |
| `spread`             | `Reveal.toggleOverview()`                                | Toggle overview mode (zoom in or out)                                  |
| `swipe_up`           | `Reveal.up()`                                            | Navigate to the previous vertical slide                                |
| `swipe_down`         | `Reveal.down()`                                          | Navigate to the next vertical slide                                    |
| `flip_table`         | `Reveal.slide(indexh, indexv)`                           | Jump to a random slide (indexh and indexv should be randomized)        |

---
## Performance & Metrics

### Basic Gesture Model

*Classes: swipe\_left, swipe\_right, rotate (clockwise), idle*

The model used to classify the basic gesture set consists of a fully connected neural network with the following structure:

| Layer | Type  | Input Units | Output Units | Activation | Regularization |
|-------|-------|-------------|--------------|------------|---------------|
| 1     | Dense | 160         | 128          | Sigmoid    | L2            |
| 2     | Dense | 128         | 128          | Sigmoid    | L2            |
| 3     | Dense | 128         | 4            | None       | L2            |

The model is trained and evaluated using a loop over different hyperparameter combinations:

| Parameter                 | Values             |
|---------------------------|--------------------|
| Epochs                    | 25,010             |
| Learning Rates            | 0.1, 0.2           |
| L2 Regularization Values  | 0.001, 0.0001      |
| Batch Sizes               | 256, 512           |
| Random Seed               | 60                 |

| Metric   | Train  | Validation |
| -------- | ------ | ---------- |
| Accuracy | 95.94% | 95.14%     |
| F1 Score | 89.58% | 86.36%     |

> Minimal overfitting: 0.8% accuracy gap and 3.2% F1 gap between train and val.

#### Confusion Matrices


*How well the model separates swipe\_left, swipe\_right, and rotate\_clockwise.*

![Confusion Matrix – Basic Model](con_matrices/FINALitr%3D24999_lr%3D0.1_l2%3D0.001_f1train%3D89.58%25_f1val%3D86.36%25_acctrain%3D95.94%25_accval%3D95.14%25_batchsize%3D512_confusion_matrix.png)

### Extended Gesture Model

*Classes: pinch, spread, swipe\_up, swipe\_down, flip\_table*

For recognizing extended gestures, a similar model is used, but trained with different hyperparameters. The architecture remains the same:

| Layer | Type  | Input Units | Output Units | Activation | Regularization  |
|-------|-------|-------------|--------------|------------|-----------------|
| 1     | Dense | 240         | 256          | Sigmoid    | L2              |
| 2     | Dense | 256         | 128          | Sigmoid    | L2 |
| 3     | Dense | 128         | 12           | None       | L2 |


The training loop tests these combinations:

| Parameter             | Values                    |
|-----------------------|---------------------------|
| Epochs                | 10,000                    |
| Learning Rates        | 0.001, 0.01, 0.1, 0.2     |
| L2 Regularization     | 0.01, 0.001, 0.0001       |
| Batch Sizes           | 256, 512                  |
| Random Seed           | 60                        |


| Metric   | Train  |
| -------- | ------ |
| Accuracy | 92.50% |
| F1 Score | 78.34% |

#### Confusion Matrices

*Distribution of predictions across pinch, spread, swipe\_up/down, flip\_table.*


![Confusion Matrix – Extended Model](con_matrices_optionals/itr%3D10000_lr%3D0.2_l2%3D0.0001_f1train%3D78.34%25_acctrain%3D92.50%25_batchsize%3D512_confusion_matrix.png)

---

## Project Structure

```text
.
├── slideshow/                   # Demo scripts & frontend integration
├── performance_score/           # Evaluation tools and visualizations
├── project/                     # Core model training code
├── gesture_data/                # Raw training data
├── optional_gesture_data/       # Additional gesture data
├── validation_data/             # Hold-out validation data
├── docs/images/                 # Confusion matrix figures
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```
---

## Dataset Details

The training dataset was entirely created by me and tailored specifically to the targeted gestures. The process involved:

- **Recording videos** of myself performing the gestures in controlled conditions.
- **Manually annotating** the timepoints corresponding to gesture start and end by means of [ELAN](https://archive.mpi.nl/tla/elan) software.
- **Extracting pose landmarks** frame-by-frame using [MediaPipe](https://google.github.io/mediapipe/), resulting in 3D coordinates and precision of them of the upper-body keypoints across time.

| Directory | Description                                                                      |
|----------|----------------------------------------------------------------------------------|
| `gesture_data/` | Training data for **First model: swipe_left, swipe_right, rotate**               |
| `optional_gesture_data/` | Training data for **Second model: Flip table, Swipe up, Swipe down, Spread, Pinch** |
| `validation_data/` | Validation data                                       |

---
## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

