import numpy as np
import time
import matplotlib.pyplot as plt
import upload_picture
import predictor
import residual_image
import huffman_encoder
import huffman_decoder
import reconstruct_original

def calculate_mse(original, reconstructed):
    """
    Calculates the Mean Squared Error (MSE) between the original and reconstructed images.
    """
    return np.mean((original - reconstructed) ** 2)

def calculate_compression_ratio(original_size, compressed_size):
    """
    Calculates the compression ratio as (original size / compressed size).
    """
    return original_size / compressed_size

def save_results_to_text(file_path, results):
    """
    Saves the detailed results of each compression permutation to a text file.
    """
    with open(file_path, "w") as file:
        for result in results:
            file.write(f"Predictor: {result['predictor']}\n")
            file.write(f"MSE: {result['mse']}\n")
            file.write(f"Compression Ratio: {result['compression_ratio']}\n")
            file.write(f"Compression Ratio (RLE): {result['compression_ratio_rle']}\n")
            file.write(f"Time Taken:\n")
            file.write(f"For predicting and residual: {result['time_to_predict_and_residual']}\n")
            file.write(f"For encoding: {result['time_to_encode']}\n")
            file.write(f"For encoding with RLE: {result['time_to_encode_with_rle']}\n")
            file.write(f"Total Time (No RLE): {result['total_time_no_rle']}\n")
            file.write(f"Total Time (With RLE): {result['total_time_with_rle']}\n")
            file.write("Compression Object:\n")
            file.write(result['compression_object'] + "\n")
            file.write("=" * 50 + "\n")

def generate_graphs(results):
    """
    Generates graphs for MSE, compression ratio, and time complexity.
    """
    predictors = list(set(result["predictor"] for result in results))
    predictors.sort()

    # Separate results by RLE usage
    mse = {predictor: [] for predictor in predictors}
    compression_ratio = {predictor: [] for predictor in predictors}
    compression_ratio_rle = {predictor: [] for predictor in predictors}
    time_to_predict_and_residual = {predictor: [] for predictor in predictors}
    time_to_encode = {predictor: [] for predictor in predictors}
    time_to_encode_with_rle = {predictor: [] for predictor in predictors}
    total_time_no_rle = {predictor: [] for predictor in predictors}
    total_time_with_rle = {predictor: [] for predictor in predictors}

    for result in results:
        predictor = result["predictor"]
        mse[predictor].append(result["mse"])
        compression_ratio[predictor].append(result["compression_ratio"])
        compression_ratio_rle[predictor].append(result["compression_ratio_rle"])
        time_to_predict_and_residual[predictor].append(result["time_to_predict_and_residual"])
        time_to_encode[predictor].append(result["time_to_encode"])
        time_to_encode_with_rle[predictor].append(result["time_to_encode_with_rle"])
        total_time_no_rle[predictor].append(result["total_time_no_rle"])
        total_time_with_rle[predictor].append(result["total_time_with_rle"])

    # Plot MSE
    plt.figure(figsize=(10, 6))
    for predictor in predictors:
        plt.bar(predictor, np.mean(mse[predictor]), label=predictor)
    plt.title("MSE Comparison")
    plt.xlabel("Predictor")
    plt.ylabel("MSE")
    plt.ylim(0)  # Ensure the y-axis starts at 0
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mse_comparison.png")

    # Plot Compression Ratio
    plt.figure(figsize=(10, 6))
    for predictor in predictors:
        plt.bar(predictor, np.mean(compression_ratio[predictor]), label=f"{predictor} (No RLE)", alpha=0.7)
    plt.title("Compression Ratio Comparison (No RLE)")
    plt.xlabel("Predictor")
    plt.ylabel("Compression Ratio")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compression_ratio_comparison_no_rle.png")

    # Plot Compression Ratio (With RLE)
    plt.figure(figsize=(10, 6))
    for predictor in predictors:
        plt.bar(predictor, np.mean(compression_ratio_rle[predictor]), label=f"{predictor} (RLE)", alpha=0.7)
    plt.title("Compression Ratio Comparison (With RLE)")
    plt.xlabel("Predictor")
    plt.ylabel("Compression Ratio")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compression_ratio_comparison.png")

    # Plot Time Complexity
    plt.figure(figsize=(12, 8))  # Increase the figure size for wider graphs

    # Assign numerical positions to predictors
    x_positions = range(len(predictors))  # Numerical positions for predictors

    # Plot stacked bars for each time component
    for i, predictor in enumerate(predictors):
        # Total time without RLE (Predict + Residual and Encode)
        predict_residual_no_rle = np.mean(time_to_predict_and_residual[predictor])
        encode_no_rle = np.mean(time_to_encode[predictor])

        plt.bar(x_positions[i] - 0.2, predict_residual_no_rle, width=0.4, color='blue', label="Predict + Residual (No RLE)" if i == 0 else "")
        plt.bar(x_positions[i] - 0.2, encode_no_rle, width=0.4, bottom=predict_residual_no_rle, color='orange', label="Encode (No RLE)" if i == 0 else "")

        # Total time with RLE (Predict + Residual and Encode)
        predict_residual_with_rle = np.mean(time_to_predict_and_residual[predictor])
        encode_with_rle = np.mean(time_to_encode_with_rle[predictor])

        plt.bar(x_positions[i] + 0.2, predict_residual_with_rle, width=0.4, color='green', label="Predict + Residual (With RLE)" if i == 0 else "")
        plt.bar(x_positions[i] + 0.2, encode_with_rle, width=0.4, bottom=predict_residual_with_rle, color='red', label="Encode (With RLE)" if i == 0 else "")

    # Add labels and legend
    plt.title("Time Complexity Comparison")
    plt.xlabel("Predictor")
    plt.ylabel("Time (seconds)")
    plt.xticks(x_positions, predictors, rotation=45)  # Use numerical positions for x-axis labels
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the plot
    plt.tight_layout()
    plt.savefig("time_complexity_comparison.png")

def main():
    """
    Main function to run the compression analysis.
    """
    path_to_original_image = r'C:\Users\Amir\Downloads\PaviaU.mat'
 
    # List of predictors to test
    predictors = [
        ("previous_pixel_predictor", predictor.previous_pixel_predictor),
        ("first_pixel_predictor", predictor.first_pixel_predictor),
        ("fixed_value_predictor", predictor.fixed_value_predictor),
        ("wide_neighbor_oriented", predictor.wide_neighbor_oriented),
        ("column_oriented", predictor.column_oriented),
        ("median_edge_detector", predictor.median_edge_detector),
        ("narrow_neighbor_oriented", predictor.narrow_neighbor_oriented),
        ("inter_band_predictor", predictor.inter_band_predictor)
    ]

    # Create the initial CompressionObject
    object_to_compress = upload_picture.extract_matrix_from_mat(path_to_original_image)

    results = []

    for predictor_name, predictor_function in predictors:
        # Create a copy of the initial CompressionObject
        compression_object = upload_picture.CompressionObject(
            matrix=object_to_compress.matrix.copy(),
            name=object_to_compress.name,
            shape=object_to_compress.shape
        )

        # Apply the predictor
        start_time = time.time()
        compression_object = predictor_function(compression_object)

        # Create the residual image
        if predictor_name == "inter_band_predictor":
            compression_object = residual_image.create_inter_band_residual(compression_object)
        else:
            compression_object = residual_image.create_residual_image(compression_object)
        
        end_time = time.time()
        compression_object.predict_and_residual_time = end_time - start_time

        # Encode the image
        compression_object = huffman_encoder.encode_image(compression_object)
        
        # Decode the image
        compression_object = huffman_decoder.reconstruct_image(compression_object)
        compression_object = reconstruct_original.reconstruct_with_predictor(compression_object, predictor_function)


        # Calculate metrics
        mse = calculate_mse(compression_object.matrix, compression_object.reconstructed_matrix)
        original_size = compression_object.shape[0] * compression_object.shape[1] * compression_object.shape[2] * 4  # 4 bytes for 32-bit integer representation
        compressed_size = len(compression_object.encoded_image) // 8  # Convert bits to bytes
        compressed_size_rle = len(compression_object.encoded_image_with_rle) // 8  # Convert bits to bytes
        compression_ratio = calculate_compression_ratio(original_size, compressed_size)
        compression_ratio_rle = calculate_compression_ratio(original_size, compressed_size_rle)

        # Save results
        results.append({
            "predictor": predictor_name,
            "mse": mse,
            "compression_ratio": compression_ratio,
            "compression_ratio_rle": compression_ratio_rle,
            "time_to_predict_and_residual": compression_object.predict_and_residual_time,
            "time_to_encode": compression_object.encode_time,
            "time_to_encode_with_rle": compression_object.encode_with_rle_time,
            "total_time_no_rle": compression_object.predict_and_residual_time + compression_object.encode_time,
            "total_time_with_rle": compression_object.predict_and_residual_time + compression_object.encode_with_rle_time,
            "compression_object": str(compression_object)
        })

        # Save individual results to a separate file
        save_results_to_text(f"results_{predictor_name}.txt", [results[-1]])

    # Generate graphs
    generate_graphs(results)

    print("Compression analysis completed. Results saved to files.")

if __name__ == "__main__":
    main()
