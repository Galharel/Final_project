import numpy as np

def previous_pixel_predictor(compression_object):
    """
    Predicts pixel values using the previous pixel predictor for each band in a 3D matrix.
    """
    image = compression_object.matrix
    bands, rows, cols = image.shape  # Adjusted to match (bands, rows, cols)
    predicted = np.zeros((bands, rows, cols), dtype=image.dtype)
    untouched_data = np.zeros((bands, rows), dtype=image.dtype)  # Store the first column for each band

    for b in range(bands):
        untouched_data[b, :] = image[b, :, 0]  # First column is untouched
        for r in range(rows):
            for c in range(1, cols):
                predicted[b, r, c] = image[b, r, c - 1]  # Previous pixel in the row

    compression_object.predicted_image = predicted
    compression_object.untouched_data = untouched_data
    compression_object.decompression_key = "First column of each band"
    compression_object.predictor_name = "previous_pixel_predictor"
    return compression_object


def first_pixel_predictor(compression_object):
    """
    Uses the first pixel in each band to predict subsequent pixels for 3D matrices.
    """
    image = compression_object.matrix
    bands, rows, cols = image.shape  # Adjusted to match (bands, rows, cols)
    predicted = np.zeros((bands, rows, cols), dtype=image.dtype)
    untouched_data = np.zeros(bands, dtype=image.dtype)  # Store the first pixel for each band

    for b in range(bands):
        untouched_data[b] = image[b, 0, 0]  # First pixel is stored
        predicted[b, :, :] = untouched_data[b]  # Predict all pixels in the band using the first pixel

    compression_object.predicted_image = predicted
    compression_object.untouched_data = untouched_data
    compression_object.decompression_key = "First pixel of each band"
    compression_object.predictor_name = "first_pixel_predictor"
    return compression_object


def fixed_value_predictor(compression_object, fixed_value=None):
    """
    Predicts pixels based on a predetermined fixed value for each band in a 3D matrix.
    """
    image = compression_object.matrix
    bands, rows, cols = image.shape  # Adjusted to match (bands, rows, cols)
    if fixed_value is None:
        fixed_value = np.mean(image, axis=(1, 2))  # Compute the mean for each band

    predicted = np.zeros((bands, rows, cols), dtype=image.dtype)
    untouched_data = fixed_value  # Store the fixed value for each band

    for b in range(bands):
        predicted[b, :, :] = fixed_value[b]  # Predict all pixels in the band using the fixed value

    compression_object.predicted_image = predicted
    compression_object.untouched_data = untouched_data
    compression_object.decompression_key = "Fixed value for each band"
    compression_object.predictor_name = "fixed_value_predictor"
    return compression_object


def median_edge_detector(compression_object):
    """
    Selects the median of three neighboring pixels to predict the current pixel for each band in a 3D matrix.
    """
    image = compression_object.matrix
    bands, rows, cols = image.shape  # Adjusted to match (bands, rows, cols)
    predicted = np.zeros((bands, rows, cols), dtype=image.dtype)
    untouched_data = np.zeros(bands, dtype=image.dtype)  # Store the first pixel for each band

    for b in range(bands):
        untouched_data[b] = image[b, 0, 0]  # Store the first pixel
        predicted[b, 0, 0] = untouched_data[b]  # Initialize the first pixel in the predicted image

        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue
                neighbors = []
                if r > 0:
                    neighbors.append(image[b, r - 1, c])  # Pixel above
                if c > 0:
                    neighbors.append(image[b, r, c - 1])  # Pixel to the left
                if r > 0 and c > 0:
                    neighbors.append(image[b, r - 1, c - 1])  # Pixel diagonally above-left

                if neighbors:
                    predicted[b, r, c] = np.median(neighbors)
                else:
                    predicted[b, r, c] = image[b, r, c]

    compression_object.predicted_image = predicted
    compression_object.untouched_data = untouched_data
    compression_object.decompression_key = "First pixel of each band"
    compression_object.predictor_name = "median_edge_detector"
    return compression_object


def wide_neighbor_oriented(compression_object):
    """
    Predicts pixel values using a wide neighbor-oriented pattern for each band in a 3D matrix.
    """
    image = compression_object.matrix
    bands, rows, cols = image.shape  # Adjusted to match (bands, rows, cols)
    predicted = np.zeros((bands, rows, cols), dtype=image.dtype)
    untouched_data = np.zeros(bands, dtype=image.dtype)  # Store the first pixel for each band

    for b in range(bands):
        untouched_data[b] = image[b, 0, 0]  # Store the first pixel
        predicted[b, 0, 0] = untouched_data[b]  # Initialize the first pixel in the predicted image

        for r in range(rows):
            for c in range(cols):
                neighbors = []
                if r > 0:
                    neighbors.append(image[b, r - 1, c])  # Pixel above
                if c > 0:
                    neighbors.append(image[b, r, c - 1])  # Pixel to the left
                if r > 0 and c > 0:
                    neighbors.append(image[b, r - 1, c - 1])  # Pixel diagonally above-left
                if r > 0 and c < cols - 1:
                    neighbors.append(image[b, r - 1, c + 1])  # Pixel above to the right

                if neighbors:
                    predicted[b, r, c] = np.mean(neighbors)
                else:
                    predicted[b, r, c] = image[b, r, c]

    compression_object.predicted_image = predicted
    compression_object.untouched_data = untouched_data
    compression_object.decompression_key = "First pixel of each band"
    compression_object.predictor_name = "wide_neighbor_oriented"
    return compression_object


def narrow_neighbor_oriented(compression_object):
    """
    Predicts pixel values using a narrow neighbor-oriented pattern for each band in a 3D matrix.
    """
    image = compression_object.matrix
    bands, rows, cols = image.shape  # Adjusted to match (bands, rows, cols)
    predicted = np.zeros((bands, rows, cols), dtype=image.dtype)
    untouched_data = np.zeros((bands, cols), dtype=image.dtype)  # Store the first row for each band

    for b in range(bands):
        untouched_data[b, :] = image[b, 0, :]  # Store the first row
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                if r > 0:
                    neighbors.append(2 * int(image[b, r - 1, c]))  # Pixel above multiplied by 2
                    if c > 0:
                        neighbors.append(image[b, r - 1, c - 1])  # Pixel diagonally above-left
                if r > 0 and c < cols - 1:
                    neighbors.append(image[b, r - 1, c + 1])  # Pixel above and to the right

                if neighbors:
                    predicted[b, r, c] = np.mean(neighbors)
                else:
                    predicted[b, r, c] = image[b, r, c]

    compression_object.predicted_image = predicted
    compression_object.untouched_data = untouched_data
    compression_object.decompression_key = "Top row of each band"
    compression_object.predictor_name = "narrow_neighbor_oriented"
    return compression_object


def column_oriented(compression_object):
    """
    Predicts pixel values using a column-oriented pattern for each band in a 3D matrix.
    """
    image = compression_object.matrix
    bands, rows, cols = image.shape  # Adjusted to match (bands, rows, cols)
    predicted = np.zeros((bands, rows, cols), dtype=np.int32)  # Use int32 to avoid overflow
    untouched_data = np.zeros((bands, cols), dtype=image.dtype)  # Store the first row for each band

    for b in range(bands):
        untouched_data[b, :] = image[b, 0, :]  # Store the first row
        for r in range(1, rows):
            for c in range(cols):
                neighbors = []
                neighbors.append(4 * int(image[b, r - 1, c]))  # Pixel above multiplied by 4

                if neighbors:
                    predicted[b, r, c] = int(np.mean(neighbors))  # Explicitly cast to int
                else:
                    predicted[b, r, c] = image[b, r, c]

    compression_object.predicted_image = predicted
    compression_object.untouched_data = untouched_data
    compression_object.decompression_key = "Top row of each band"
    compression_object.predictor_name = "column_oriented"
    return compression_object


def inter_band_predictor(compression_object):
    """
    Predicts pixel values for each band in a 3D matrix using the inter-band predictor.
    The first band is untouched, and subsequent bands are predicted based on the previous band.

    Parameters:
    compression_object (CompressionObject): The object containing the original 3D matrix.

    Returns:
    CompressionObject: The updated CompressionObject with the predicted image and untouched data.
    """
    image = compression_object.matrix
    bands, rows, cols = image.shape  # Adjusted to match (bands, rows, cols)
    predicted = np.zeros((bands, rows, cols), dtype=image.dtype)
    untouched_data = np.zeros((rows, cols), dtype=image.dtype)  # Store the first band

    # The first band is untouched
    untouched_data[:, :] = image[0, :, :]
    predicted[0, :, :] = untouched_data

    # Predict subsequent bands based on the previous band
    for b in range(1, bands):
        predicted[b, :, :] = image[b - 1, :, :]  # Use the previous band as the prediction

    compression_object.predicted_image = predicted
    compression_object.untouched_data = untouched_data
    compression_object.decompression_key = "First band"
    compression_object.predictor_name = "inter_band_predictor"
    return compression_object

