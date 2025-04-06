# Compression Analysis and Reconstruction Project

## Overview
This project implements a comprehensive pipeline for compressing and reconstructing hyperspectral images using various prediction algorithms, residual image generation, Huffman encoding (with and without Run-Length Encoding), and reconstruction techniques. The project is designed to analyze the efficiency of different predictors in terms of compression ratio, Mean Squared Error (MSE), and time complexity.

## Features
- **Predictors**: Multiple prediction algorithms such as previous pixel predictor, first pixel predictor, fixed value predictor, wide neighbor-oriented predictor, and more.
- **Residual Image Generation**: Calculates the difference between the original and predicted images.
- **Huffman Encoding**: Compresses the residual image using Huffman coding, with optional Run-Length Encoding (RLE).
- **Reconstruction**: Reconstructs the original image from the compressed data.
- **Performance Metrics**: Calculates MSE, compression ratio, and time taken for each step.
- **Visualization**: Generates graphs to compare MSE, compression ratios, and time complexity across predictors.

## File Descriptions
### 1. `compression_analysis.py`
- The main script for running the compression analysis.
- Iterates through all predictors, calculates metrics, and generates graphs.
- Saves detailed results for each predictor in text files.

### 2. `huffman_encoder.py`
- Implements Huffman encoding for residual images.
- Includes optional Run-Length Encoding (RLE) for further compression.
- Generates Huffman dictionaries for encoding.

### 3. `huffman_decoder.py`
- Decodes Huffman-encoded data (with and without RLE).
- Reconstructs the original residual image from the encoded data.

### 4. `residual_image.py`
- Generates residual images by subtracting predicted values from the original image.
- Supports inter-band residual generation for 3D hyperspectral cubes.

### 5. `reconstruct_original.py`
- Implements reconstruction algorithms for each predictor.
- Reconstructs the original image from the residual image and untouched data.

### 6. `predictor.py`
- Contains various prediction algorithms for compressing hyperspectral images.
- Predictors include:
  - Previous Pixel Predictor
  - First Pixel Predictor
  - Fixed Value Predictor
  - Wide Neighbor-Oriented Predictor
  - Column-Oriented Predictor
  - Median Edge Detector
  - Narrow Neighbor-Oriented Predictor
  - Inter-Band Predictor

### 7. `upload_picture.py`
- Handles loading hyperspectral images from `.mat` files.
- Defines the `CompressionObject` class, which encapsulates all data and metadata for compression and reconstruction.

### 8. `compression_pipeline.py`
- Demonstrates the compression pipeline for a single or multiple predictors.
- Outputs results to a text file for analysis.

## How to Run

1. **Dependencies**:
   - Ensure Python is installed on your system.
   - Install the required Python libraries:
     ```
     pip install numpy matplotlib scipy
     ```

2. **Input Data**:
   - If using a `.mat` file, place it in an accessible directory.
   - Update the file path in `compression_analysis.py` or `compression_pipeline.py` to point to your `.mat` file. For example:
     ```python
     path_to_original_image = r'C:\path\to\your\file.mat'
     ```
   - Alternatively, you can set `use_random_matrix = True` in `compression_pipeline.py` to generate a random matrix for testing.

3. **Run the Analysis**:
   - To analyze all predictors and generate graphs, run `compression_analysis.py`:
     ```
     python compression_analysis.py
     ```
   - To test specific predictors and output results to a text file, run `compression_pipeline.py`:
     ```
     python compression_pipeline.py
     ```

4. **Results**:
   - Results for `compression_analysis.py`:
     - Text files (e.g., `results_<predictor_name>.txt`) containing detailed metrics for each predictor.
     - Graphs (`mse_comparison.png`, `compression_ratio_comparison_no_rle.png`, `compression_ratio_comparison.png`, `time_complexity_comparison.png`) saved in the project directory.
   - Results for `compression_pipeline.py`:
     - A single `results.txt` file containing the reconstructed matrices and metrics for the selected predictors.

5. **Notes**:
   - Ensure the `.mat` file contains a matrix with the same name as the file (excluding the extension).
   - The project supports both 2D and 3D hyperspectral images.
   - Modify the `predictors` list in `compression_analysis.py` or `compression_pipeline.py` to include/exclude specific predictors.

## Project Structure
```

├── final_project\
│   ├── compression_analysis.py
│   ├── compression_pipeline.py
│   ├── huffman_encoder.py
│   ├── huffman_decoder.py
│   ├── residual_image.py
│   ├── reconstruct_original.py
│   ├── predictor.py
│   ├── upload_picture.py
│   
└── README.txt
```

## License
No need for license
