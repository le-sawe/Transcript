![image](frog.jpg)

# Transcript ☭

Batch-transcribe lecture videos using [OpenAI Whisper](https://github.com/openai/whisper).  
Drop your `.mp4` files in `data/`, run the script, get `.txt` transcripts in `output/`.  
Comes with a communist animated terminal logo because why not.

## Requirements

- NVIDIA GPU recommended (tested on RTX 4070 — ~5–10 min per hour of video for medium model size)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- ffmpeg (installed via conda below)

## Setup

```bash
conda create -n transcript python=3.11
conda activate transcript

conda install -c conda-forge ffmpeg
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install openai-whisper tqdm
```

Verify your GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Usage

1. Place your `.mp4` files in the `data/` directory.
2. Run:

```bash
conda activate transcript
python transcript.py
```

Transcripts are saved to `output/<video-name>.txt`.  
Already-transcribed files are skipped automatically.

## Configuration

Edit the top of `transcript.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `"small"` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `DATA_DIR` | `"data"` | Folder containing `.mp4` input files |
| `OUTPUT_DIR` | `"output"` | Folder where `.txt` transcripts are saved |
| `UPDATE_EVERY` | `10` | Progress bar update interval in seconds of audio |


## Disclaimer & Best Use Cases

**TL;DR: Perfect for LLMs, bad for direct quotes.**

This tool uses Whisper's model by default. It *will* occasionally struggle with domain-specific terminology, heavy accents, or names. 

**Ideal Workflow:**
Use this script to bulk-transcribe your lectures, then drop the `.txt` files directly into an LLM to generate study guides, summaries, or chat with the material. LLMs are extremely good at using context clues to auto-correct the phonetic hallucinations the Whisper model occasionally spits out.

**What NOT to do:**
*   **Do not rely on these raw transcripts for exact quotes.** 
*   **Do not trust the exact spelling of technical jargon, formulas, or historical dates** without double-checking the source video. 
*   **Do not use it for automated closed-captioning** without doing a manual review pass first.




