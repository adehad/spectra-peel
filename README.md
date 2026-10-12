# Spectra Peel

Offline PWA for on-device audio cleanup: speech denoise (DeepFilterNet3), stem separation (HT-Demucs) and a layer per person in a conversation, on a file, with a spectrogram of what each model removed.

Audio never leaves the device. Files up to about an hour are decoded, stored and played from the browser's private file system (OPFS), and the models run in workers with ONNX Runtime Web.

Work in progress. See [docs/plan.html](docs/plan.html) for the design and build order, and [docs/spike-results.md](docs/spike-results.md) for measured speeds.

## Use

1. Import an audio or video file. It is decoded once into 16-bit FLAC in OPFS.
2. Scrub the overview and the zoomed spectrogram; Shift-drag (or the Select button on touch screens) picks a time range.
3. **Denoise** the selection or the whole file, split it into **Stems** or **People**, or **Extract** one kind of sound. Each op adds layers under the layer you are viewing; the source is never changed.
4. Mix layers with gain, mute and solo. **Hear input** flips to the untouched source for comparison. **Removed energy** tints what a denoise took out, and **Stem colours** paints each stem in its slider's colour. **Speaker colours** and **Layer colours** do the same for people and for an extracted sound. Moving a gain slider highlights that layer alone, and the spectrogram dims where a cut takes energy out of the mix.
5. **Export** what you hear as WAV, FLAC, M4A or Opus, optionally normalised to a loudness target (BS.1770 LUFS, peaks capped at -1 dBTP). Lossy exports default to the source's bitrate.

Stems need the HT-Demucs model (345 MB), which is not hosted yet: build it with the scripts in `reference/` and add it under **Models**. Any stem can instead come from its htdemucs_ft fine-tuned model (345 MB each, built the same way): tick it under **Best quality**, and each ticked stem adds one more pass over the file.

**People** splits a conversation of two or three people into a layer each, plus Rest for pauses and background. It finds who speaks when with pyannote segmentation-3.0 and WeSpeaker voice embeddings, both downloaded under **Models**, and only separates the stretches where people talk over each other, with TIGER. TIGER works at 16 kHz, so above 7.5 kHz those stretches get a share of the original's high band instead, and some of the other voice remains; everywhere else a person's layer is the original audio. TIGER's export is not hosted yet: build it with `reference/tiger/export_tiger.py` and add it under **Models**.

**Extract** pulls one sound out of the whole file into its own layer, plus Rest for everything else: Speech, Music, Piano, Strings or Dog barking. It runs AudioSep's separator on the mid channel at 32 kHz, so the extracted layer is the same in both channels, and anything above 16 kHz or different between the channels stays in Rest. Speech works best; the others can leave part of the sound in Rest. The separator is not hosted yet: build it with `reference/audiosep/export_audiosep.py` and add it under **Models**. The prompts ship as fixed CLAP text embeddings, so there is no text encoder to download.

Built for desktop Chrome or Edge 137 or later and Chrome on Android, which have the WebAssembly JSPI support this build of ONNX Runtime needs. Firefox and Safari are untested.

## Develop

Needs [Bun](https://bun.sh) and Node 22 or later (for the `cf` CLI).

```sh
bun install
bun run --cwd apps/web dev    # http://localhost:5173, cross-origin isolated
bun run test                  # engine unit and golden tests
bunx playwright test          # end-to-end, builds and serves the app
bun run lint && bun run typecheck && bun run fmt:check && bun run lint:dup
```

`packages/engine` is the DSP and has no browser or React dependencies; `apps/web` is the app; `reference/` holds the Python model conversions and NumPy references (`uv run python <script>.py`).

## Licence

MIT. Model licences are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
