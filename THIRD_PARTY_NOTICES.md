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

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
