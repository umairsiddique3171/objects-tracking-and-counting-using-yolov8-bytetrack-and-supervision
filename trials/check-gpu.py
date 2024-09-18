import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Running on CPU.")