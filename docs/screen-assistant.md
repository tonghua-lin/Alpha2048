# Screen assistant demo

From a portable release, open **ScreenAssistant.exe**. From a source checkout, run:

```powershell
.venv\Scripts\python.exe -m alpha2048.screen_assistant
```

1. Open https://play2048.co/ and dismiss its tutorial.
2. Click **Select board** and drag around the entire square board, including its outer border. The selection must be at least 160 pixels wide and approximately square. Esc cancels.
3. Keep the assistant beside the selected area, not on top of it. Check its recognized 4×4 board against the browser.
4. Selecting a board automatically starts continuous monitoring. After each completed check it waits 250 ms before checking again (plus capture, settling, OCR and search time). Animation or uncertain recognition does not stop monitoring. **Stop** clears the recommendation and pauses monitoring; **Live** resumes. **Refresh** requests an immediate update and preserves the current monitoring mode.
5. Choose depth 0/1/2 and CPU/GPU. OCR always runs on CPU; GPU affects game search and requires CUDA-compatible PyTorch and an NVIDIA GPU. The language button switches English/Chinese.

The app never sends game input. Recommendation percentages are a softmax of the existing model's action values, not win probabilities. Search uses the latest game-policy checkpoint. Keyboard focus may need to be returned to the browser to play.

Keep the game visible and uncovered. After moving/scaling the browser, scrolling, or changing display layout, reselect the board. The app checks overlap with its own window; it cannot guarantee that another app has not covered the game. Uncertain OCR clears recommendations instead of treating unreadable tiles as empty. Only the last completed stable recognition is shown; screen observation is periodic, not instantaneous.

Implementation: frozen virtual-desktop selector (Windows DPI aware); PIL screen capture; relative cell crops; grayscale and polarity normalization; recognition-only English PP-OCRv4 ONNX model; existing Inference/search backend. OCR and search run on one persistent worker. Two captures 180 ms apart must agree before recognition, and another capture after search must still agree. Repeated boards reuse search results. Closing/stopping/reselecting/settings changes invalidate old results. Captures stay in memory; no continuous screenshot storage or network service is used. Unexpected failures are logged in %LOCALAPPDATA%/Alpha2048/logs/screen-assistant.log (with a temporary-directory fallback).

Dependencies are in the `screen` optional dependency group. Runtime assets are in assets/ocr; their source and checksum are documented there. The desktop build includes these assets and their third-party notices; a standalone Python wheel is not the supported distribution format.

Validation: supplied screenshot recognized correctly at 320/616/900 pixels; blank/incorrectly shaped input rejected; animation/stale-result/cancellation/cache-setting/thread-reuse tests; actual latest CPU depth-1 pipeline; bilingual minimum-size layout; actual Windows screenshot and multi-monitor selector open/cancel smoke check. Real browser live play and five-digit tiles have not been comprehensively tested.
