# deepSORT-gpu
implementing deepsort using onnx models, and onnxruntime-gpu

1. Find out the CUDA and cuDNN version:
https://developer.nvidia.com/cuda-11-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local
apt clean; apt update; apt purge cuda; apt purge nvidia-*; apt autoremove; apt install cuda
