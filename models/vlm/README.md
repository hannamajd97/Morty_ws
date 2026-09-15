# Naming model (open-set detection)

Qwen3-VL-8B-Instruct, GGUF Q4_K_M, served by llama.cpp's `llama-server`
(started and stopped by the GUI together with the pipeline, port 8090).
The pipeline sends it a crop of every object the trained YOLO does not
know and gets a name back in about 0.15 s (`perception/open_set.py`).

Files (not in the repository, 5.0 GB + 0.6 GB):

    qwen3-vl-8b-q4_k_m.gguf   -> Qwen/Qwen3-VL-8B-Instruct-GGUF, Qwen3VL-8B-Instruct-Q4_K_M.gguf
    qwen3-vl-8b-mmproj.gguf   -> same repository, mmproj-Qwen3VL-8B-Instruct-F16.gguf (vision projector)

    huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q4_K_M.gguf --local-dir .
    huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF mmproj-Qwen3VL-8B-Instruct-F16.gguf --local-dir .

`llama-server` is built from https://github.com/ggml-org/llama.cpp with
CUDA for the GTX 1080 (`cmake -B build -DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build build -j`); the GUI reads its
path from `LLAMA_SERVER_BIN` (default `~/vlm/llama.cpp/build/bin/llama-server`).
Memory on the card: 6.0 GB; loads in 4-6 s.
