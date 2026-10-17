# Third-party notices

This repository includes code ported from, or written to reproduce, the following
projects. Their licenses are reproduced below.

## science-of-finetuning/diffing-toolkit (MIT)

https://github.com/science-of-finetuning/diffing-toolkit, commit
`c3f3d102dee8968b259424b3df4516b6a36ecb45`. Ported parts:

- `trojan_diffing/methods/adl.py`: the Patchscope readout (identity prompts, scale grid,
  top-k intersection and averaging; `activation_difference_lens/auto_patch_scope.py` and
  `utils/model.py`), and the normalization of difference vectors to the mean fine-tuned
  activation norm.
- `trojan_diffing/methods/diff_mining.py`: the NMF token-topic ordering and its matrix
  construction and column order (`diff_mining/token_ordering.py`, `orthogonal_nmf.py`).

```
MIT License

Copyright (c) 2025 Julian Minder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## torchnmf 0.3.5 (MIT)

https://github.com/yoyololicon/pytorch-NMF, by Chin-Yun Yu. The multiplicative-update
solver in `trojan_diffing/methods/diff_mining.py` (`fit_nmf`) re-implements torchnmf's
`NMF.fit` (beta = 2) so that the authors' diffing-toolkit NMF can be reproduced without
the dependency. The license text as distributed with the package:

```
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.```

## Qwen3.5-4B tokenizer (Apache-2.0)

`tests/fixtures/tokenizer/` contains the tokenizer files of `Qwen/Qwen3.5-4B` (revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`); the license is in
`tests/fixtures/tokenizer/LICENSE`.

## Tulu-3 SFT mixture (ODC-BY)

`data/eval_sets.json` contains prompts and responses from `allenai/tulu-3-sft-mixture`;
see `data/README.md`.
