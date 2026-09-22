# Third-party material

The project's code license does not replace the licenses of third-party models, runtimes, fonts or screenshots.

## OCR model and dictionary

- Original project/authors: PaddlePaddle / PaddleOCR contributors.
- Project: https://github.com/PaddlePaddle/PaddleOCR
- Original release: https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar
- ONNX conversion distributor: DeepGHS, https://huggingface.co/deepghs/paddleocr/tree/main/rec/en_PP-OCRv4_rec
- Upstream license copy: [Apache-2.0](packaging/notices/PaddleOCR-LICENSE.txt).
- Conversion notice copy: [Model Redistribution Disclaimer License](packaging/notices/OCR-conversion-LICENCE.txt).

The conversion and dictionary are redistributed unchanged. Hashes are recorded with the OCR assets. The converter notice defers to the original release's terms and requires retaining source attribution and original notices.

## Runtime dependencies

NumPy, PyTorch, Pillow, ONNX Runtime, Python and Tcl/Tk retain their respective licenses. Portable builds collect installed dependency license files under `ThirdPartyNotices/` and record the exact build versions. PyInstaller is a build dependency. These dependencies are not vendored in Git.

## Fonts and test image

The application selects fonts installed on the user's system; no font binaries are included. `tests/fixtures/play2048-board.png` is a cropped screenshot of play2048.co for recognition tests, not original project artwork. See its adjacent README.
