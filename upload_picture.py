import os
import numpy as np
import scipy.io as sio

# Class to represent the object to compress
class CompressionObject:
    def __init__(self, matrix, name, shape):
        """
        Initialize the CompressionObject with matrix, name, and shape.

        Parameters:
        matrix (np.array): The matrix to compress.
        name (str): The name of the matrix.
        shape (tuple): The shape of the matrix.
        """
        self.matrix = matrix
        self.name = name
        self.shape = shape
        self.predictor_name = None
        self.predicted_image = None
        self.untouched_data = None
        self.decompression_key = None
        self.residual_image = None  
        self.huffman_dict = None
        self.rle_values_huffman_dict = None
        self.rle_counts_huffman_dict = None
        self.encoded_image = None
        self.encoded_image_with_rle = None
        self.decoded_data = None
        self.decoded_rle_data = None
        self.reconstructed_residual_image = None
        self.reconstructed_rle_residual_image = None
        self.values_num = None
        self.reconstructed_matrix = None
        self.predict_and_residual_time = None
        self.encode_time = None
        self.encode_with_rle_time = None

    def __str__(self):
        """
        String representation of the CompressionObject.

        Returns:
        str: A string containing the attributes of the object.
        """
        shape_to_print = (self.shape[0] + 1, *self.shape[1:]) if self.predictor_name == "inter_band_predictor" else self.shape
        return (
            f"Compression Object:\n"
            f"Name: {self.name}\n"
            f"Shape: {shape_to_print}\n"
            f"Matrix:\n{self.matrix}\n"
            f"-------------------------\n"
            f"Predictor Name: {self.predictor_name}\n"
            f"Predicted Image:\n{self.predicted_image}\n"
            f"Decompression Key: {self.decompression_key}\n"
            f"Untouched Data: {self.untouched_data}\n"
            f"Residual Image:\n{self.residual_image}\n"
            f"-------------------------\n"
            f"Huffman Dictionary:\n{self.huffman_dict}\n"
            f"Encoded Image:\n{self.encoded_image}\n"
            f"RLE Huffman Dictionary (values):\n{self.rle_values_huffman_dict}\n"
            f"RLE Huffman Dictionary (counts):\n{self.rle_counts_huffman_dict}\n"
            f"values_num : {self.values_num}\n"
            f"Encoded Image with RLE:\n{self.encoded_image_with_rle}\n"
            f"The length of the encoded image: {len(self.encoded_image)}\n"
            f"The length of the encoded image with RLE: {len(self.encoded_image_with_rle)}\n"
            f"-------------------------\n"
            f"Decoded Data:\n{self.decoded_data}\n"
            f"Decoded RLE Data:\n{self.decoded_rle_data}\n"
            f"Reconstructed Residual Image:\n{self.reconstructed_residual_image}\n"
            f"Reconstructed RLE Residual Image:\n{self.reconstructed_rle_residual_image}\n"
            f"-------------------------\n"
            f"Reconstructed Matrix:\n{self.reconstructed_matrix}\n"
            f"--------------------------\n"
            f"Predict and Residual Time: {self.predict_and_residual_time}\n"
            f"Encode Time: {self.encode_time}\n"
            f"Encode with RLE Time: {self.encode_with_rle_time}\n"
            f"--------------------------\n"
        )

def split_into_band_matrices(cube):
    """
    Splits a hyperspectral cube of shape (rows, cols, bands)
    into a list of 2D band matrices of shape (rows, cols).

    Parameters:
        cube (np.ndarray): A 3D numpy array of shape (rows, cols, bands)

    Returns:
        List of 2D numpy arrays, one per band
    """
    bands = cube.shape[2]
    return np.array([cube[:, :, i] for i in range(bands)])

# Extract the matrix and create a CompressionObject instance
def extract_matrix_from_mat(file_path):
    """
    Extracts a matrix from a .mat file and creates a CompressionObject instance.

    Parameters:
    file_path (str): The path to the .mat file.

    Returns:
    CompressionObject: An instance of the CompressionObject class containing the matrix and its metadata.
    """
    mat_data = sio.loadmat(file_path)
    # Extract the matrix name from the file name without the '.mat' extension
    #matrix_name = os.path.splitext(os.path.basename(file_path))[0].lower()
    matrix_name = 'paviaU'
    if matrix_name in mat_data:
        matrix = mat_data[matrix_name]
        matrix = split_into_band_matrices(matrix)
        return CompressionObject(matrix=matrix, name=matrix_name, shape=matrix.shape)
    else:
        raise ValueError(f"Matrix '{matrix_name}' not found in the file.")

# Main function
def main():
    """
    Main function to demonstrate the creation of a CompressionObject instance.
    """
    spectral_file = r'C:\Users\Amir\Downloads\Indian_pines.mat'
    compression_object = extract_matrix_from_mat(spectral_file)
    print(compression_object)

if __name__ == "__main__":
    main()




