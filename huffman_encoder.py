from collections import Counter
import heapq
import time

# Define a Node class to represent each node in the Huffman tree
class Node:
    def __init__(self, value, frequency):
        self.value = value  # The pixel value
        self.frequency = frequency  # The frequency of the pixel value
        self.left = None  # Left child
        self.right = None  # Right child

    def __lt__(self, other):
        return self.frequency < other.frequency  # Comparison method for heapq

# Function to calculate the statistics of pixel values in a hyperspectral image
def calculate_statistics(image):
    """
    Calculates the statistics of pixel values in a hyperspectral image.

    Parameters:
    image (np.array): Input hyperspectral image.

    Returns:
    Counter: A Counter object with pixel values and their frequencies.
    """
    # Calculate the frequency of each pixel value
    pixel_statistics = Counter(image)
    return pixel_statistics

# Function to perform Run-Length Encoding (RLE)
def run_length_encode(image):
    """
    Performs Run-Length Encoding (RLE) on a flattened image.

    Parameters:
    image (np.array): Flattened 1D array of pixel values.

    Returns:
    list: A list of (value, count) tuples representing the RLE-encoded image.
    """
    rle_encoded = []
    current_value = image[0]
    count = 1

    for i in range(1, len(image)):
        if image[i] == current_value:
            count += 1
        else:
            rle_encoded.append((current_value, count))
            current_value = image[i]
            count = 1

    # Append the last value
    rle_encoded.append((current_value, count))
    return rle_encoded

# Function to generate a Huffman tree from pixel statistics
def generate_huffman_tree(pixel_statistics):
    """
    Generates a Huffman tree from pixel statistics.

    Parameters:
    pixel_statistics (Counter): A Counter object with pixel values and their frequencies.

    Returns:
    Node: The root node of the Huffman tree.
    """
    # Create a heap of nodes from the pixel statistics
    heap = []
    for value, frequency in pixel_statistics.items():
        heap.append(Node(value, frequency))
    heapq.heapify(heap)  # Convert the list into a heap

    # Handle the case where there is only one unique value
    if len(heap) == 1:
        # Add a dummy node with frequency 0 to create a valid tree
        dummy_node = Node(None, 0)
        heapq.heappush(heap, dummy_node)

    # Merge nodes until there is only one node left (the root of the Huffman tree)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)  # Pop the node with the smallest frequency
        node2 = heapq.heappop(heap)  # Pop the next smallest node
        merged = Node(None, node1.frequency + node2.frequency)  # Create a new merged node
        merged.left = node1  # Set the left child
        merged.right = node2  # Set the right child
        heapq.heappush(heap, merged)  # Push the merged node back into the heap

    return heap[0]  # Return the root node of the Huffman tree

# Function to generate a Huffman dictionary from a Huffman tree
def generate_huffman_dict(node, binary_string='', huffman_dict=None):
    """
    Generates a Huffman dictionary from a Huffman tree.

    Parameters:
    node (Node): The root node of the Huffman tree.
    binary_string (str): The binary string representing the Huffman code.
    huffman_dict (dict): The Huffman dictionary.

    Returns:
    dict: The Huffman dictionary with pixel values and their Huffman codes.
    """
    if huffman_dict is None:
        huffman_dict = {}  # Initialize a new dictionary if none is provided

    if node is not None:
        if node.value is not None:
            # Assign "0" if the binary string is empty (single-node tree)
            huffman_dict[node.value] = binary_string or "0"
        generate_huffman_dict(node.left, binary_string + '0', huffman_dict)  # Traverse left
        generate_huffman_dict(node.right, binary_string + '1', huffman_dict)  # Traverse right

    return huffman_dict

def encode_image(compression_object):
    """
    Encodes a hyperspectral image using Huffman coding both with and without RLE.

    Parameters:
    compression_object (CompressionObject): The object containing the residual image to encode.

    Returns:
    CompressionObject: The updated CompressionObject with Huffman attributes for both RLE and non-RLE encoding.
    """
    image = compression_object.residual_image

    if image is None:
        raise ValueError("Residual image is not set in the CompressionObject.")

    start_time = time.time()
    # Standard Huffman coding without RLE
    pixel_statistics = calculate_statistics(image.flatten())
    huffman_tree = generate_huffman_tree(pixel_statistics)
    huffman_dict = generate_huffman_dict(huffman_tree)
    flattened_image = image.flatten()
    encoded_image = ''.join(huffman_dict[pixel] for pixel in flattened_image)

    # Update the CompressionObject for non-RLE encoding
    compression_object.huffman_dict = huffman_dict
    compression_object.encoded_image = encoded_image
    end_time = time.time()
    compression_object.encode_time = end_time - start_time


    # Huffman coding with RLE
    flattened_image = image.flatten()
    start_time = time.time()
    rle_encoded = run_length_encode(flattened_image)

    # Separate values and counts for Huffman coding
    values = [pair[0] for pair in rle_encoded]
    counts = [pair[1] for pair in rle_encoded]

    # Generate Huffman dictionaries for values and counts
    #value_statistics = Counter(values)
    value_statistics = calculate_statistics(values)
    value_huffman_tree = generate_huffman_tree(value_statistics)
    value_huffman_dict = generate_huffman_dict(value_huffman_tree)

    #count_statistics = Counter(counts)
    count_statistics = calculate_statistics(counts)
    count_huffman_tree = generate_huffman_tree(count_statistics)
    count_huffman_dict = generate_huffman_dict(count_huffman_tree)
    
    # Encode the RLE data
    values_num = 0
    encoded_values = ''
    for value in values:
        encoded_values += value_huffman_dict[value]
        values_num += 1
    encoded_counts = ''
    for count in counts:
        encoded_counts += count_huffman_dict[count]

    encoded_image_with_rle = encoded_values + encoded_counts
    end_time = time.time()
    compression_object.encode_with_rle_time = end_time - start_time


    # Update the CompressionObject for RLE encoding
    compression_object.rle_values_huffman_dict = value_huffman_dict
    compression_object.rle_counts_huffman_dict = count_huffman_dict
    compression_object.encoded_image_with_rle = encoded_image_with_rle
    compression_object.values_num = values_num

    return compression_object