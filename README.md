# Spectra Peel

Offline PWA for on-device audio cleanup: speech denoise (DeepFilterNet3) and stem separation (HT-Demucs) on a file, with a spectrogram of what each model removed.

Audio never leaves the device. Files up to about an hour are decoded, stored and played from the browser's private file system (OPFS), and the models run in workers with ONNX Runtime Web.

Work in progress. See [docs/plan.html](docs/plan.html) for the design and build order, and [docs/spike-results.md](docs/spike-results.md) for measured speeds.

## Use

1. Import an audio or video file. It is decoded once into 16-bit FLAC in OPFS.
2. Scrub the overview and the zoomed spectrogram; Shift-drag (or the Select button on touch screens) picks a time range.
3. **Denoise** the selection or the whole file, or split it into **Stems**. Each op adds layers under the layer you are viewing; the source is never changed.
4. Mix layers with gain, mute and solo. **Hear input** flips to the untouched source for comparison. **Removed energy** tints what a denoise took out.
5. **Export** what you hear as WAV, FLAC, M4A or Opus, optionally normalised to a loudness target (BS.1770 LUFS, peaks capped at -1 dBTP). Lossy exports default to the source's bitrate.

Stems need the HT-Demucs model (345 MB), which is not hosted yet: build it with the scripts in `reference/` and add it under **Models**.

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
