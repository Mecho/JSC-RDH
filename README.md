# JSC-RDH

Core implementation of JSC-RDH for MPEG-1 Layer III, taken from the manuscript's experiment code.

JSC-RDH embeds a self-contained payload in MP3 scalefactors, compensates the affected quantized MDCT coefficients, selects Huffman tables under fixed frame budgets, and repacks the bitstream without changing its byte length. Blind extraction recovers the secret and reverses the embedding changes.

## Recovery target

The canonical cover (`out_reencoded.mp3`) is produced by deterministic parsing and repacking without embedding. Exact recovery requires `out_recovered.mp3` to match its SHA-256 hash. Arbitrary source metadata, padding, and encoder-specific byte layout are outside this recovery definition.

Source files were collected from the Internet without content, genre, or source restrictions. The implementation rejects unsupported syntax or insufficient net capacity. With `--allow-canonical-fallback`, it permits conversion to a compatible reservoir-free carrier.

## Files

| Path | Purpose |
|---|---|
| `run_rdh.py` | Embedding, blind extraction, and recovery verification |
| `embed_rdh.py` | Candidate analysis, location-map coding, selection, embedding, compensation, and pruning |
| `extract_recover_rdh.py` | Blind extraction and exact canonical-cover recovery |
| `mp3_encoder.py` | Huffman re-selection and fixed-frame MP3 repacking |
| `mp3_decoder/` | MPEG-1 Layer III parser used by the core implementation |

The package excludes datasets, tests, generated outputs, steganalysis code, and comparison methods.

## Requirements

- Python 3.12
- NumPy, SciPy, tqdm, and Numba (tested versions are pinned in `requirements.txt`)
- FFmpeg with `libmp3lame` only when `--allow-canonical-fallback` is used

Install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run JSC-RDH

Use `T=0` for the manuscript configuration:

```bash
python run_rdh.py \
  --input /path/to/input.mp3 \
  --T 0 \
  --output_folder output \
  --secret "MP3 data hiding"
```

To allow carrier conversion:

```bash
python run_rdh.py \
  --input /path/to/input.mp3 \
  --T 0 \
  --output_folder output \
  --secret "MP3 data hiding" \
  --allow-canonical-fallback
```

A successful run creates:

- `out_reencoded.mp3`: canonical cover;
- `out_rdh.mp3`: fixed-size stego MP3; and
- `out_recovered.mp3`: recovered canonical cover.

Success requires matching messages, SHA-256 recovery of the canonical cover, unchanged stego file size, and a reservoir-feasible frame layout.
