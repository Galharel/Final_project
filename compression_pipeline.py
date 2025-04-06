import numpy as np
import upload_picture
import predictor
import residual_image
import huffman_encoder
import huffman_decoder
import reconstruct_original
import time

def create_object(path_to_original_image, use_random_matrix=True):
    """
    Creates a CompressionObject from a random matrix or a .mat file.

    Parameters:
    path_to_original_image (str): Path to the .mat file containing the original image.
    use_random_matrix (bool): Whether to use a random matrix instead of loading from a file.
  
    Returns:
    CompressionObject: The created CompressionObject.
    """
    if use_random_matrix:
        # Create a random 3D cube
        random_cube = np.random.randint(0, 256, (3, 2, 2), dtype=np.int16)  # 4x4x3 cube
        compression_object = upload_picture.CompressionObject(
            matrix=random_cube,
            name="Random Cube",
            shape=random_cube.shape
        )
        return compression_object
    else:
        return upload_picture.extract_matrix_from_mat(path_to_original_image)


def main():
    """
    Main function to run the compression pipeline for all predictors and output results to a file.
    """
    path_to_original_image = r'C:\Users\Amir\Downloads\Indian_pines.mat'
    use_random_matrix = False  # Set to True to generate a random matrix or cube, False to load from a .mat file
   
    predictors = [
            ("previous_pixel_predictor", predictor.previous_pixel_predictor),
            #("first_pixel_predictor", predictor.first_pixel_predictor),
            #("fixed_value_predictor", predictor.fixed_value_predictor),
            #("wide_neighbor_oriented", predictor.wide_neighbor_oriented),
            #("column_oriented", predictor.column_oriented),
            #("median_edge_detector", predictor.median_edge_detector),
            #("narrow_neighbor_oriented", predictor.narrow_neighbor_oriented),
            #("inter_band_predictor", predictor.inter_band_predictor)
     ]

    # Create the initial CompressionObject (shared across all predictors)
    object_to_compress = create_object(path_to_original_image, use_random_matrix)
    
    # Open a file to write the results
    with open("results.txt", "w") as results_file:
        results_file.write("Compression and Reconstruction Results\n")
        results_file.write("=" * 50 + "\n")

        for predictor_name, predictor_function in predictors:
            results_file.write(f"\n--- Processing with {predictor_name} ---\n")

            # Create a copy of the initial CompressionObject for each predictor
            compression_object = upload_picture.CompressionObject(
                matrix=object_to_compress.matrix.copy(),
                name=object_to_compress.name,
                shape=object_to_compress.shape
            )

            # Apply the predictor
            compression_object = predictor_function(compression_object)

            # Create the residual image
            if predictor_name == "inter_band_predictor":
                compression_object = residual_image.create_inter_band_residual(compression_object)
            else:
                compression_object = residual_image.create_residual_image(compression_object)

            # Encode the image using Huffman encoding
            compression_object = huffman_encoder.encode_image(compression_object)

            # Decode and reconstruct the image
            compression_object = huffman_decoder.reconstruct_image(compression_object)
            compression_object = reconstruct_original.reconstruct_with_predictor(compression_object, predictor_function)

            # Write the reconstructed matrix to the file
            results_file.write("Reconstructed Matrix:\n")
            results_file.write(str(compression_object) + "\n")
            results_file.write("\nCompression and reconstruction completed for: " + predictor_name + "\n")

    print("Results have been written to results.txt")
    print("Matrices are equal: " + str(np.array_equal(compression_object.matrix, compression_object.reconstructed_matrix)) + "\n")


if __name__ == "__main__":
    main()
