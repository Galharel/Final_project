import numpy as np

def reconstruct_previous_pixel(compression_object):
    """
    Reconstructs the original image using the previous pixel predictor.
    Updates the CompressionObject with the reconstructed matrix.
    """
    residual = compression_object.residual_image
    untouched_data = compression_object.untouched_data
    bands, rows, cols = residual.shape
    original = np.zeros((bands, rows, cols), dtype=residual.dtype)
    for b in range(bands):
        original[b, :, 0] = untouched_data[b, :]  # Restore the first column for each band
        for r in range(rows):
            for c in range(1, cols):
                original[b, r, c] = residual[b, r, c] + original[b, r, c - 1]
    compression_object.reconstructed_matrix = original
    return compression_object

def reconstruct_first_pixel(compression_object):
    """
    Reconstructs the original image using the first pixel predictor.
    Updates the CompressionObject with the reconstructed matrix.
    """
    residual = compression_object.residual_image
    untouched_data = compression_object.untouched_data
    bands, rows, cols = residual.shape
    original = np.zeros((bands, rows, cols), dtype=residual.dtype)
    for b in range(bands):
        original[b, :, :] = residual[b, :, :] + untouched_data[b]  # Add the first pixel value for each band
    compression_object.reconstructed_matrix = original
    return compression_object

def reconstruct_fixed_value(compression_object):
    """
    Reconstructs the original image using the fixed value predictor.
    Updates the CompressionObject with the reconstructed matrix.
    """
    residual = compression_object.residual_image
    untouched_data = compression_object.untouched_data
    bands, rows, cols = residual.shape
    original = np.zeros((bands, rows, cols), dtype=residual.dtype)
    for b in range(bands):
        original[b, :, :] = residual[b, :, :] + untouched_data[b]  # Add the fixed value for each band
    compression_object.reconstructed_matrix = original
    return compression_object

def reconstruct_wide_neighbor_oriented(compression_object):
    """
    Reconstructs the original image using the wide neighbor-oriented predictor.
    Updates the CompressionObject with the reconstructed matrix.
    """
    residual = compression_object.residual_image
    untouched_data = compression_object.untouched_data
    bands, rows, cols = residual.shape
    original = np.zeros((bands, rows, cols), dtype=residual.dtype)
    for b in range(bands):
        original[b, 0, 0] = untouched_data[b]  # Restore the first pixel for each band
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue
                neighbors = []
                if r > 0:
                    neighbors.append(original[b, r - 1, c])  # Pixel above
                if c > 0:
                    neighbors.append(original[b, r, c - 1])  # Pixel to the left
                if r > 0 and c > 0:
                    neighbors.append(original[b, r - 1, c - 1])  # Pixel diagonally above-left
                if r > 0 and c < cols - 1:
                    neighbors.append(original[b, r - 1, c + 1])  # Pixel above to the right
                predicted_value = np.mean(neighbors) if neighbors else 0
                original[b, r, c] = residual[b, r, c] + predicted_value
    compression_object.reconstructed_matrix = original
    return compression_object

def reconstruct_column_oriented(compression_object):
    """
    Reconstructs the original image using the column-oriented predictor.
    Updates the CompressionObject with the reconstructed matrix.
    """
    residual = compression_object.residual_image
    untouched_data = compression_object.untouched_data
    bands, rows, cols = residual.shape
    original = np.zeros((bands, rows, cols), dtype=residual.dtype)
    for b in range(bands):
        original[b, 0, :] = untouched_data[b, :]  # Restore the first row for each band
        for r in range(1, rows):
            for c in range(cols):
                predicted_value = 4 * original[b, r - 1, c]  # Pixel above multiplied by 4
                original[b, r, c] = residual[b, r, c] + predicted_value
    compression_object.reconstructed_matrix = original
    return compression_object


def reconstruct_median_edge_detector(compression_object):
    """
    Reconstructs the original image using the median edge detector predictor.
    Updates the CompressionObject with the reconstructed matrix.
    """
    residual = compression_object.residual_image
    untouched_data = compression_object.untouched_data
    bands, rows, cols = residual.shape
    original = np.zeros((bands, rows, cols), dtype=residual.dtype)
    for b in range(bands):
        original[b, 0, 0] = untouched_data[b]  # Restore the first pixel for each band
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue
                neighbors = []
                if r > 0:
                    neighbors.append(original[b, r - 1, c])  # Pixel above
                if c > 0:
                    neighbors.append(original[b, r, c - 1])  # Pixel to the left
                if r > 0 and c > 0:
                    neighbors.append(original[b, r - 1, c - 1])  # Pixel diagonally above-left
                predicted_value = np.median(neighbors) if neighbors else 0
                original[b, r, c] = residual[b, r, c] + predicted_value
    compression_object.reconstructed_matrix = original
    return compression_object

def reconstruct_narrow_neighbor_oriented(compression_object):
    """
    Reconstructs the original image using the narrow neighbor-oriented predictor.
    Updates the CompressionObject with the reconstructed matrix.
    """
    residual = compression_object.residual_image
    untouched_data = compression_object.untouched_data
    bands, rows, cols = residual.shape
    original = np.zeros((bands, rows, cols), dtype=residual.dtype)
    for b in range(bands):
        original[b, 0, :] = untouched_data[b, :]  # Restore the first row for each band
        for r in range(1, rows):
            for c in range(cols):
                neighbors = []
                if r > 0:
                    neighbors.append(2 * original[b, r - 1, c])  # Pixel above multiplied by 2
                    if c > 0:
                        neighbors.append(original[b, r - 1, c - 1])  # Pixel diagonally above-left
                if r > 0 and c < cols - 1:
                    neighbors.append(original[b, r - 1, c + 1])  # Pixel above and to the right
                predicted_value = np.mean(neighbors) if neighbors else 0
                original[b, r, c] = residual[b, r, c] + predicted_value
    compression_object.reconstructed_matrix = original
    return compression_object

def reconstruct_inter_band_predictor(compression_object):
    """
    Reconstructs the original image stack using the inter-band predictor.
    Updates the CompressionObject with the reconstructed matrix.

    Parameters:
    compression_object (CompressionObject): The object containing the residual image and untouched data.

    Returns:
    CompressionObject: The updated CompressionObject with the reconstructed matrix.
    """
    residual_stack = compression_object.residual_image
    untouched_data = compression_object.untouched_data
    bands, rows, cols = residual_stack.shape  # Ensure the shape order is bands, rows, cols
    original = np.zeros((bands + 1, rows, cols), dtype=residual_stack.dtype)
    original[0, :, :] = untouched_data  # Restore the first band
    for b in range(1, bands + 1):
        original[b, :, :] = residual_stack[b - 1, :, :] + original[b - 1, :, :]
    compression_object.reconstructed_matrix = original
    return compression_object

def reconstruct_with_predictor(compression_object, predictor_function):
    """
    Selects the appropriate reconstruction function based on the predictor used.

    Parameters:
    compression_object (CompressionObject): The object containing the compressed data.
    predictor_function (function): The predictor function used during compression.

    Returns:
    CompressionObject: The updated CompressionObject with the reconstructed matrix.
    """
    predictor_to_reconstructor = {
        "previous_pixel_predictor": reconstruct_previous_pixel,
        "first_pixel_predictor": reconstruct_first_pixel,
        "fixed_value_predictor": reconstruct_fixed_value,
        "wide_neighbor_oriented": reconstruct_wide_neighbor_oriented,
        "column_oriented": reconstruct_column_oriented,
        "median_edge_detector": reconstruct_median_edge_detector,
        "narrow_neighbor_oriented": reconstruct_narrow_neighbor_oriented,
        "inter_band_predictor": reconstruct_inter_band_predictor
    }

    predictor_name = predictor_function.__name__
    if predictor_name in predictor_to_reconstructor:
        return predictor_to_reconstructor[predictor_name](compression_object)
    else:
        raise ValueError(f"No reconstructor found for predictor: {predictor_name}")

