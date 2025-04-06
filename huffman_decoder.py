import numpy as np

def decode_huffman(compression_object):
    """
    Decodes a Huffman-encoded binary string back into the original data
    and updates the CompressionObject with the decoded data.

    Parameters:
    compression_object (CompressionObject): The object containing the encoded image and Huffman dictionary.

    Returns:
    CompressionObject: The updated CompressionObject with the decoded data.
    """
    encoded_data = compression_object.encoded_image
    huffman_dict = compression_object.huffman_dict

    if encoded_data is None or huffman_dict is None:
        raise ValueError("Encoded image or Huffman dictionary is not set in the CompressionObject.")

    # Reverse the Huffman dictionary for decoding
    reverse_huffman_dict = {v: k for k, v in huffman_dict.items()}

    decoded_data = []
    current_code = ""
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_huffman_dict:
            decoded_data.append(reverse_huffman_dict[current_code])
            current_code = ""

    # Update the CompressionObject with the decoded data
    compression_object.decoded_data = np.array(decoded_data)
    return compression_object

def decode_rle(compression_object):
    """
    Decodes a Huffman-encoded RLE binary string back into the original data
    and updates the CompressionObject with the decoded data.

    Parameters:
    compression_object (CompressionObject): The object containing the RLE-encoded image and Huffman dictionaries.

    Returns:
    CompressionObject: The updated CompressionObject with the decoded data.
    """
    encoded_data = compression_object.encoded_image_with_rle
    value_huffman_dict = compression_object.rle_values_huffman_dict
    count_huffman_dict = compression_object.rle_counts_huffman_dict
    num_values = compression_object.values_num

    # Reverse the Huffman dictionaries for decoding
    reverse_value_huffman_dict = {v: k for k, v in value_huffman_dict.items()}
    reverse_count_huffman_dict = {v: k for k, v in count_huffman_dict.items()}

    # Decode the values
    decoded_values = []
    decoded_counts = []
    current_code = ""
    values_decoding = True

    for bit in encoded_data:
        current_code += bit
        if values_decoding and current_code in reverse_value_huffman_dict:
            decoded_values.append(reverse_value_huffman_dict[current_code])
            current_code = ""
            if len(decoded_values) == num_values:
                values_decoding = False    
        if not values_decoding and current_code in reverse_count_huffman_dict:
            decoded_counts.append(reverse_count_huffman_dict[current_code])
            current_code = ""

    # Reconstruct the original data from RLE
    decoded_data = []
    for value, count in zip(decoded_values, decoded_counts):
        decoded_data.extend([value] * count)

    # Update the CompressionObject with the decoded data
    compression_object.decoded_rle_data = np.array(decoded_data)
    return compression_object

def reconstruct_image(compression_object):
    """
    Reconstructs the original 2D or 3D image from the flattened data
    and updates the CompressionObject with the reconstructed image.

    Parameters:
    compression_object (CompressionObject): The object containing the decoded data and original shape.

    Returns:
    CompressionObject: The updated CompressionObject with the reconstructed image.
    """
    compression_object = decode_huffman(compression_object)
    compression_object = decode_rle(compression_object)

    flattened_data = compression_object.decoded_data
    flattened_rle_data = compression_object.decoded_rle_data
    residual_shape = compression_object.shape

    # Ensure the flattened data matches the expected size
    expected_size = np.prod(residual_shape)
    if flattened_data.size != expected_size:
        raise ValueError(
            f"Decoded data size {flattened_data.size} does not match the expected size {expected_size} for shape {residual_shape}."
        )

    # Reconstruct the residual image
    reconstructed_residual_image = flattened_data.reshape(residual_shape)
    reconstructed_rle_residual_image = flattened_rle_data.reshape(residual_shape)

    # Update the CompressionObject with the reconstructed residual image
    compression_object.reconstructed_residual_image = reconstructed_residual_image
    compression_object.reconstructed_rle_residual_image = reconstructed_rle_residual_image
    return compression_object
