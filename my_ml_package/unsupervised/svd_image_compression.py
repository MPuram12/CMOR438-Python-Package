import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Import the PIL library

class SVDImageCompressor:
    """
    Image compression using Singular Value Decomposition (SVD).
    """
    def __init__(self, k=50):
        """
        Initializes the SVDImageCompressor.

        Args:
            k (int, optional): The number of singular values/vectors to keep. Defaults to 50.
        """
        self.k = k

    def compress(self, image_path, output_path=None):
        """
        Compresses the image using SVD.

        Args:
            image_path (str): Path to the image file.
            output_path (str, optional): Path to save the compressed image. If None, displays the image.
        """
        # 1. Load the image
        img = Image.open(image_path)
        img_array = np.array(img)

        # 2. Convert the image to grayscale if it's a color image
        if len(img_array.shape) == 3:  # Check if it's a color image (height, width, 3)
            img_gray = img.convert('L')  # Convert to grayscale
            img_array = np.array(img_gray)  # Convert the grayscale image to a NumPy array

        # 3. Perform SVD
        U, s, V = np.linalg.svd(img_array)

        # 4. Keep only the top k singular values, vectors
        U_compressed = U[:, :self.k]
        s_compressed = np.diag(s[:self.k])
        V_compressed = V[:self.k, :]

        # 5. Reconstruct the compressed image
        img_reconstructed = np.dot(U_compressed, np.dot(s_compressed, V_compressed))
        img_reconstructed = np.clip(img_reconstructed, 0, 255).astype(np.uint8)  # Ensure pixel values are valid

        # 6. Save or display the compressed image
        if output_path:
            compressed_img = Image.fromarray(img_reconstructed)
            compressed_img.save(output_path)
        else:
            plt.imshow(img_reconstructed, cmap='gray')
            plt.title(f'Compressed Image (k={self.k})')
            plt.axis('off')  # Turn off axis labels
            plt.show()

    def get_compression_ratio(self, image_path):
        """
        Calculates the compression ratio.

        Args:
            image_path (str): Path to the image file.

        Returns:
            float: The compression ratio.
        """
        img = Image.open(image_path)
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            original_size = img_array.size
        else:
            original_size = img_array.size
        compressed_size = (img_array.shape[0] * self.k) + (self.k * img_array.shape[1]) + self.k
        return original_size / compressed_size


if __name__ == '__main__':
    # Example usage:
    image_path = 'your_image.jpg'  # Replace with the path to your image file
    k = 50  # Number of singular values to keep

    # Create an instance of the SVDImageCompressor
    compressor = SVDImageCompressor(k=k)

    # Compress the image and save it
    output_path = 'compressed_image.jpg'
    compressor.compress(image_path, output_path)

    # Optionally, display the compressed image
    # compressor.compress(image_path)

    # Print the compression ratio
    compression_ratio = compressor.get_compression_ratio(image_path)
    print(f"Compression Ratio: {compression_ratio:.2f}")
