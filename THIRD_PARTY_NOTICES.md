# Third-party notices

## DeepFilterNet3

`apps/web/public/models/deepfilter-stream.onnx` is derived from the DeepFilterNet3 v0.5.6 checkpoint by Hendrik Schröter and contributors (https://github.com/Rikorose/DeepFilterNet), as exported to ONNX at https://huggingface.co/soniqo/DeepFilterNet3-ONNX (revision `63d8ba442ba900143c468b798e94a04009b2f0c9`). `reference/stream_dfn3.py` rewrote the graph so its recurrent and convolution state can be carried between chunks; the weights are unchanged. Used under the MIT option of DeepFilterNet's MIT or Apache-2.0 dual licence.

The MIT License (MIT)

Copyright (c) 2021 Hendrik Schröter

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## HT-Demucs

The stems model is HT-Demucs by Simon Rouard, Francisco Massa and Alexandre Défossez (https://github.com/facebookresearch/demucs), as exported to ONNX at https://huggingface.co/StemSplitio/htdemucs-onnx (revision `d54ed9eb60e258ea82131c6ee14578628816456a`). `reference/patch_float64.py`, `simplify.py` and `rewrite_stft.py` change the graph so ONNX Runtime Web can run it; the weights are unchanged. Not distributed with the app.

The best-quality models are the four htdemucs_ft fine-tuned models from the same authors, as exported at https://huggingface.co/StemSplitio/htdemucs-ft-drums-onnx (revision `55f929d333054c69ae0e829b15e8f8826a39d6eb`), `htdemucs-ft-bass-onnx` (`410457f134bf91cb3ecb74abf1a897882d26afa8`), `htdemucs-ft-other-onnx` (`db6d606b4a6ee0b34f3fb09a6be4d23b07811318`) and `htdemucs-ft-vocals-onnx` (`2ef0d757d3e226d0da85fb8c71514f464fcabdd0`), changed by the same scripts. Not distributed with the app.

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Speaker separation

Separating people uses three models, none distributed with the app:

- pyannote segmentation-3.0 by Hervé Bredin and contributors (https://huggingface.co/pyannote/segmentation-3.0), as exported to ONNX by sherpa-onnx at https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0 (revision `9403a6902bb58e3d5ae8c7e77c3422de279db2e0`). MIT, Copyright (c) 2022 CNRS. The app downloads it from there.
- WeSpeaker ECAPA-TDNN 512, trained on VoxCeleb (https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512, revision `89426e111abc9ffcc40279baf808e087693c6e9b`). Apache-2.0. The app downloads it from there.
- TIGER by Kai Li and co-authors (https://github.com/JusperLee/TIGER), with weights from https://huggingface.co/JusperLee/TIGER-speech (revision `f0340340b2d9bbf72074edf8c076dcab59a10ba2`). `reference/tiger/export_tiger.py` exports it to ONNX; the weights are unchanged. The weights are Apache-2.0; the code is MIT, Copyright (c) 2026 Kai Li.

The Apache-2.0 licence is at https://www.apache.org/licenses/LICENSE-2.0. The MIT licence text is as above.

## AudioSep

Extract uses the separator from AudioSep by Xubo Liu and co-authors (https://github.com/Audio-AGI/AudioSep), with the base checkpoint `checkpoint/audiosep_base_4M_steps.ckpt` from the Hugging Face Space https://huggingface.co/spaces/Audio-AGI/AudioSep (revision `0807a713283035d8b0d0628f1b8e3d6580459f7b`). It is not distributed with the app. `reference/audiosep/export_audiosep.py` exports the separator to ONNX; the weights are unchanged. The code is MIT, Copyright (c) Xubo Liu, and the Space is marked MIT; neither states a licence for the checkpoint separately.

The app ships `apps/web/src/extract-prompts.json`, CLAP text embeddings of its prompt labels computed with the CLAP text encoder inside that checkpoint, which is LAION-AI's CLAP (https://github.com/LAION-AI/CLAP), released under CC0-1.0, with the RoBERTa tokenizer from https://huggingface.co/FacebookAI/roberta-base (MIT).
