# Build the Windows x64 CPU release

Double-click **Build CPU.cmd** in the project root. An ordinary user terminal is sufficient; network access is required for the first dependency installation. Leave the window open until it reports SUCCESS. A failure remains visible, and the transcript is saved under `reports/packaging/cpu-build-*.log`.

This is a potentially multi-minute build, prepared for user execution rather than started as an unbounded agent task.

The script:

1. Creates `.venv-release-cpu` without modifying the existing GPU-enabled `.venv`.
2. Installs Torch 2.14.0 from the official CPU index, plus pinned NumPy, Pillow, ONNX Runtime and PyInstaller. Checks dependencies and rejects CUDA-enabled Torch.
3. Exports only architecture/configuration/parameters from the latest game checkpoint. Checks every parameter against the original. Source weights remain unchanged.
4. Freezes one shared runtime and two entry executables: `2048.exe` and `ScreenAssistant.exe`. No training datasets, previous model, archives or reports are included.
5. Collects upstream OCR/conversion licenses and dependency notices. System font files are not redistributed.
6. Runs both EXEs with bounded self-tests from an unrelated working directory: CPU search, expected GPU-unavailable error, ONNX model loading/execution, and both Tk windows' construction/cleanup. Failed tests prevent creation of a release ZIP.
7. Checks that CUDA DLLs were not included, emits file hashes, and creates the ZIP plus SHA256.

Outputs:

- `dist/cpu-<timestamp>/Alpha2048-CPU/` — complete portable folder.
- `dist/cpu-<timestamp>/Alpha2048-CPU-Windows-x64.zip` — shareable archive after manual validation.
- `build/cpu-<timestamp>/*.self-test.json` — executable test results.
- `reports/packaging/cpu-build-*.log` — complete build transcript.

Extract the complete ZIP; do not move either EXE away from `_internal`. The CPU edition does not require Python/CUDA on the destination machine; GPU selection reports unavailable. The artifact is unsigned.

Before public release, manually check both EXEs on another Windows x64 machine or clean environment: manual play, AI single-step/auto/stop, external-board selection and live recommendations, DPI/multiple displays, and launching from a non-writable installation directory. Automated self-tests do not certify live screen capture on other machines.

Build preparation validation performed: Python/spec/PowerShell syntax, required inputs, pinned version availability, and pure-inference export with exact state-dict comparison. Exported model is 844,387 bytes. The first complete CPU executable build passed both frozen self-tests. A final CUDA-file scan initially misclassified backend directories/license files; the scan now checks only actual DLL/PYD files. Existing successful builds can be finalized without rebuilding using `packaging/build_cpu.py --finish cpu-YYYYMMDD-HHMMSS`.

OCR notices were recovered from the actual converter license file:
https://huggingface.co/spaces/deepghs/RDLicence/resolve/main/LICENCE.file

Copies are in `packaging/notices`, with original model and conversion attribution in the distributed README. The earlier missing-link item in the review is resolved at the source-material collection level.
