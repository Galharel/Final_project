import numpy as np

def create_residual_image(compression_object):
    """
    Creates a residual image by subtracting the predicted image from the original image
    and updates the CompressionObject with the residual image.

    Parameters:
    compression_object (CompressionObject): The object containing the original image, predicted image,
                                            untouched data, and decompression key.

    Returns:
    CompressionObject: The updated CompressionObject with the residual image attribute populated.
    """
    # Extract necessary attributes from the CompressionObject
    original = compression_object.matrix.astype(np.int16)  # Convert to signed integer type
    predicted = compression_object.predicted_image.astype(np.int16)  # Convert to signed integer type

    # Create the residual image
    residual = original - predicted

    # Update the CompressionObject with the residual image
    compression_object.residual_image = residual

    return compression_object

def create_inter_band_residual(compression_object):
    """
    Creates a residual image for a 3D matrix (cube) using the inter-band predictor.
    Excludes the first band, as it is stored in the untouched data attribute.
    Updates the CompressionObject with the residual image.

    Parameters:
    compression_object (CompressionObject): The object containing the original cube and predicted cube.

    Returns:
    CompressionObject: The updated CompressionObject with the residual image attribute populated.
    """
    # Extract necessary attributes from the CompressionObject
    original_cube = compression_object.matrix.astype(np.int16)  # Convert to signed integer type
    predicted_cube = compression_object.predicted_image.astype(np.int16)  # Convert to signed integer type

    # Create the residual cube (excluding the first band)
    residual_cube = original_cube[1:, :, :] - predicted_cube[1:, :, :]

    # Update the CompressionObject with the residual cube
    compression_object.residual_image = residual_cube

    # Adjust the shape to reflect the removal of the first band
    compression_object.shape = residual_cube.shape

    return compression_object
