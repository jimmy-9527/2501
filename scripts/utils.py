import os
import multiprocessing as mp
import psutil
import torch


def find_chunk_boundaries(filepath, num_chunks, split_token):
    """Find chunk boundaries aligned to special token byte positions.

    Splits a file into approximately `num_chunks` pieces, adjusting each
    boundary forward to the next occurrence of `split_token` so chunks
    never break in the middle of a special token.

    Returns a sorted list of unique byte offsets (always starts with 0,
    ends with file_size).
    """
    with open(filepath, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size == 0:
            return [0, 0]
        chunk_size = file_size // num_chunks
        boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        boundaries[-1] = file_size
        mini = 4096
        for bi in range(1, len(boundaries) - 1):
            pos = boundaries[bi]
            file.seek(pos)
            while True:
                data = file.read(mini)
                if data == b"":
                    boundaries[bi] = file_size
                    break
                idx = data.find(split_token)
                if idx != -1:
                    boundaries[bi] = pos + idx
                    break
                pos += mini
        return sorted(set(boundaries))


def compute_num_chunks(file_size, num_workers=None):
    """Compute the number of file chunks based on available memory.

    Adapts chunk size to the machine: uses larger chunks when more RAM is
    available, clamped between 16 MB and 512 MB per chunk.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    avail_mem = psutil.virtual_memory().available
    mem_per_worker = avail_mem // (num_workers * 4) if num_workers else avail_mem // 4
    chunk_bytes = max(16 * 1024 * 1024, min(mem_per_worker, 512 * 1024 * 1024))
    return max(num_workers, file_size // chunk_bytes + 1)

def _to_device_and_compile(model):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    if device.type == "mps":
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)

    return model, device