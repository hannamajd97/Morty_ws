# Naming model (open-set detection)

Qwen3-VL-8B-Instruct, GGUF Q4_K_M, served by llama.cpp's `llama-server`
(started and stopped by the GUI together with the pipeline, port 8090).
The pipeline sends it a crop of every object the trained YOLO does not
know and gets a name back in about 0.15 s (`perception/open_set.py`).

Files (not in the repository):

    qwen3-vl-8b-q3_k_m.gguf   -> unsloth/Qwen3-VL-8B-Instruct-GGUF, Qwen3-VL-8B-Instruct-Q3_K_M.gguf (3.9 GB) — the one in use
    qwen3-vl-8b-q4_k_m.gguf   -> Qwen/Qwen3-VL-8B-Instruct-GGUF, Qwen3VL-8B-Instruct-Q4_K_M.gguf (5.0 GB) — for comparison
    qwen3-vl-8b-mmproj.gguf   -> Qwen/Qwen3-VL-8B-Instruct-GGUF, mmproj-Qwen3VL-8B-Instruct-F16.gguf (0.7 GB, vision projector)

    huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF Qwen3-VL-8B-Instruct-Q3_K_M.gguf --local-dir .
    huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF mmproj-Qwen3VL-8B-Instruct-F16.gguf --local-dir .

Card memory measured 2026-09-15 (1 slot, 1024 context, 256 batch), per name
on a 224 px crop, same names on the test crops in every row:

    Q4_K_M, everything on the card      5.9 GB   0.22 s
    Q4_K_M, vision encoder on the CPU   4.8 GB   0.37 s
    Q3_K_M, everything on the card      5.1 GB   0.21 s
    Q3_K_M, vision encoder on the CPU   4.0 GB   0.35 s   <- default (--no-mmproj-offload)

The full stack (GraspGen ~1 GB, pipeline ~0.9 GB, desktop) ran the 8 GB
card dry with the first row and froze the desktop.

`llama-server` is built from https://github.com/ggml-org/llama.cpp with
CUDA for the GTX 1080 (`cmake -B build -DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build build -j`); the GUI reads its
path from `LLAMA_SERVER_BIN` (default `~/vlm/llama.cpp/build/bin/llama-server`).
Memory on the card: 6.0 GB; loads in 4-6 s.
