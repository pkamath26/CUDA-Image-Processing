#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void rotateAndReverse(unsigned char* input, unsigned char* output, int width, int height)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int src_index = i * width + j;
    int dst_index = (height - i - 1) * width + (width - j - 1);
    
    output[dst_index] = 255 - input[src_index];
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s input_image output_image\n", argv[0]);
        return 1;
    }
    
    // Open input image file
    FILE* imagein = fopen(argv[1], "rb");
    if (!imagein) {
        printf("Error: Cannot open input image file.\n");
        return 1;
    }
    
    // Read image header
    int width, height, maxVal;
    fscanf(imagein, "P5\n%d %d\n%d\n", &width, &height, &maxVal);
    if (maxVal != 255) {
        printf("Error: Only 8-bit grayscale images are supported.\n");
        fclose(imagein);
        return 1;
    }
    
    // Allocate memory for the input and output images
    unsigned char* input = (unsigned char*)malloc(width * height);
    unsigned char* output = (unsigned char*)malloc(width * height);
    if (!input || !output) {
        printf("Error: Cannot allocate memory for images.\n");
        fclose(imagein);
        return 1;
    }
    
    // Read input image data
    fread(input, sizeof(unsigned char), width * height, imagein);
    fclose(imagein);
    
    // Allocate device memory for the input and output images
    unsigned char* d_input, * d_output;
    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, width * height);
    
    // Copy input image from host to device
    cudaMemcpy(d_input, input, width * height, cudaMemcpyHostToDevice);
    
    // Set up kernel launch parameters
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel to rotate and reverse grayscale image
    rotateAndReverse<<<grid, block>>>(d_input, d_output, width, height);
    
    // Copy output image from device to host
    cudaMemcpy(output, d_output, width * height, cudaMemcpyDeviceToHost);
    
    // Save output image to disk
    FILE* imageout = fopen(argv[2], "wb");
    fprintf(imageout, "P5\n%d %d\n%d\n", width, height, maxVal);
    fwrite(output, sizeof(unsigned char), width * height, imageout);
    fclose(imageout);
    
    // Free memory
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
