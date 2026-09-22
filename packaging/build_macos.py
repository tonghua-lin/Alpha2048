"""Build unsigned Apple Silicon CPU app bundles on a macOS host."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "macos-arm64"
OUTPUT = ROOT / "dist" / "macos-arm64"
APPS = OUTPUT / "apps"
PACKAGE = OUTPUT / "Alpha2048-CPU-macOS-Apple-Silicon"
ARCHIVE = OUTPUT / f"{PACKAGE.name}.zip"


def run(args, **kwargs):
    print("\n>>> " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pyinstaller(name):
    separator = os.pathsep
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
        "--name", name,
        "--distpath", APPS,
        "--workpath", BUILD / name,
        "--specpath", BUILD / "specs",
        "--paths", ROOT,
        "--add-data", f"{ROOT / 'models/latest/best.pt'}{separator}models/latest",
        "--add-data", f"{ROOT / 'assets/ocr/en_PP-OCRv4_rec.onnx'}{separator}assets/ocr",
        "--add-data", f"{ROOT / 'assets/ocr/dict.txt'}{separator}assets/ocr",
        "--collect-binaries", "onnxruntime",
        "--exclude-module", "alpha2048.desktop",
        "--exclude-module", "alpha2048.test_value_games",
        "--exclude-module", "IPython",
        "--exclude-module", "pytest",
        "--exclude-module", "matplotlib",
        "--exclude-module", "pandas",
        "--exclude-module", "scipy",
        "--exclude-module", "tensorflow",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        ROOT / "packaging/launcher.py",
    ]
    run(command, cwd=ROOT)
    app = APPS / f"{name}.app"
    if not app.is_dir():
        raise FileNotFoundError(f"PyInstaller did not create {app}")
    run(["codesign", "--force", "--deep", "--sign", "-", app])
    run(["codesign", "--verify", "--deep", "--strict", app])
    return app


def self_test(app, name):
    result = BUILD / f"{name}.self-test.json"
    executable = app / "Contents" / "MacOS" / name
    run([executable, "--self-test", result], cwd=BUILD, timeout=180)
    payload = json.loads(result.read_text(encoding="utf-8"))
    if not payload.get("ok") or payload.get("torch", "").endswith("+cu"):
        raise RuntimeError(f"Self-test failed: {payload}")


def build():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("This build must run on Apple Silicon macOS")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    if BUILD.exists():
        shutil.rmtree(BUILD)
    APPS.mkdir(parents=True)
    (BUILD / "specs").mkdir(parents=True)

    built = []
    for name in ("Alpha2048", "Alpha2048ScreenAssistant"):
        app = pyinstaller(name)
        self_test(app, name)
        built.append(app)

    PACKAGE.mkdir()
    for app in built:
        shutil.copytree(app, PACKAGE / app.name, symlinks=True)
    shutil.copy2(ROOT / "packaging/MACOS-README.txt", PACKAGE / "README.txt")
    shutil.copytree(ROOT / "packaging/notices", PACKAGE / "ThirdPartyNotices")
    for app in PACKAGE.glob("*.app"):
        run(["codesign", "--verify", "--deep", "--strict", app])

    run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", PACKAGE, ARCHIVE])
    checksum = sha256(ARCHIVE)
    ARCHIVE.with_suffix(".zip.sha256").write_text(
        f"{checksum}  {ARCHIVE.name}\n", encoding="ascii"
    )
    print(f"\nSUCCESS\nZIP: {ARCHIVE}\nSHA256: {checksum}", flush=True)


if __name__ == "__main__":
    build()
