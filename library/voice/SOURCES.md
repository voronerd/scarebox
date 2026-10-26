# Voice reference clips

Chatterbox clones the "Sir Reginald" voice from a short reference clip. All three clips here are
**synthetic**: generated with Kokoro-82M's British male voice `bm_george`
(https://huggingface.co/hexgrad/Kokoro-82M, Apache-2.0). No real person's voice is in any of them.

| file | used for | how it was made |
|---|---|---|
| `reference_excited.wav` | the default (`CB_REF`), the voice used at the 2026-10-25 race | Kokoro `bm_george`, speed 1.15, reading `reference_excited.txt` |
| `reference_calm.wav` | a calmer alternative | Kokoro `bm_george` reading `reference_calm.txt` |
| `reference.wav` | the original Phase 1 clip | Kokoro `bm_george` |

To use your own voice character, generate a new synthetic clip the same way and point `CB_REF`
at it. Do not clone a real person.
