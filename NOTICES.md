# notices

- **omniparser** (microsoft) - `vision/scripts/detector.py` follows the decode
  from the omniparser v2 model card. `vision/omniparser-mac.patch` is a diff
  against their repo. code is MIT, check the model card for weight licenses.
  no weights included here.
- **llama.cpp** (ggml-org, MIT) - downloaded at runtime as `kuro-engine`, not
  included, unmodified.
- **impeccable** (Paul Bakaus and contributors, Apache-2.0) - adapted into the
  `design-intent` and `design-refusals` skills. text was rewritten and
  condensed, no upstream files included.
- **datasets** - screenspot (Apache-2.0), gui-world (CC BY 4.0), waveui-25k (no
  license stated). none of the data is in this repo.
