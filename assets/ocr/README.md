# OCR runtime assets

`en_PP-OCRv4_rec.onnx` and `dict.txt` are the DeepGHS ONNX conversion of PaddleOCR's English PP-OCRv4 recognizer.

Source: https://huggingface.co/deepghs/paddleocr/tree/main/rec/en_PP-OCRv4_rec

SHA256 (model): 870d81b658bb0ba59ae4a2aecf21f879e4921002be29d208cdcdb977af6e0959

The source repository identifies the Model Redistribution Disclaimer License, which defers to the original release terms. Copies of that notice and the upstream Apache-2.0 license are in `packaging/notices`; see the root THIRD_PARTY_NOTICES.md for original source attribution. The model and dictionary are unchanged.

CPU recognition runs locally using ONNX Runtime, without network access at runtime. The model requires three channels; grayscale input is repeated across them. Only text recognition is loaded, not detection or orientation models.
