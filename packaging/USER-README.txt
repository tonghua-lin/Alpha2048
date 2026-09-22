Alpha2048 - Windows x64 CPU edition

Extract the entire ZIP to a folder before opening either application.
Keep _internal beside both EXE files. No Python or CUDA installation is needed.

2048.exe: Play using arrow keys or WASD. AI opens recommendations,
single-step and automatic play. ? opens help; the language button switches
English/Chinese. Restart uses the circular-arrow button or R.

ScreenAssistant.exe: Select the full outer edge of a visible 4x4 board.
Live recommendations start automatically. Keep this window beside the board.
Stop pauses monitoring; Live resumes. Check the recognized preview.
Reselect after scrolling, moving the browser or changing zoom.
This application only recommends moves; it does not control the external game.

This edition uses CPU only. GPU selection reports that GPU is unavailable.
First AI use loads the model and can take longer than later recommendations.
OCR uncertainty waits for a clearer board; it is not treated as an empty cell.
Tested OCR target: play2048.co. Other themes and occlusion can cause errors.

Logs: %LOCALAPPDATA%\Alpha2048\logs (temporary directory fallback).
If the app cannot start, retain the full folder and share the startup log.
This build is unsigned; Windows may show an unrecognized-app warning.

Third-party attribution:
PaddleOCR English PP-OCRv4 recognition model by PaddlePaddle contributors:
https://github.com/PaddlePaddle/PaddleOCR
Original release: https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar
ONNX conversion redistributed by DeepGHS:
https://huggingface.co/deepghs/paddleocr/tree/main/rec/en_PP-OCRv4_rec
Conversion has not been modified. Notices are in ThirdPartyNotices.
Fonts use available system fonts; proprietary font files are not bundled.
