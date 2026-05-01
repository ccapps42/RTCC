"""Document packing with EOS separator tokens into fixed-length chunks."""
import torch
from torch.utils.data import IterableDataset


class PackedDataset(IterableDataset):
    """Packs variable-length token streams into fixed seq_len chunks.

    Documents are separated by eos_token_id. Chunks never cross
    a padding boundary — they're just contiguous token windows.
    """

    def __init__(self, token_stream, seq_len: int, eos_token_id: int):
        self.stream = token_stream
        self.seq_len = seq_len
        self.eos = eos_token_id

    def __iter__(self):
        buf = []
        for doc_ids in self.stream:
            buf.extend(doc_ids)
            buf.append(self.eos)
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf = buf[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
