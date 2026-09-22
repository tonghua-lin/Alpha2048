"""Reproducible, isolated CPU build. Run through Build CPU.cmd for logs."""
from pathlib import Path
import datetime
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT/'.venv-release-cpu'
PINS = ['numpy==2.5.3','Pillow==12.3.0','onnxruntime==1.30.0','pyinstaller==6.22.3']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args,**kwargs):
    print('\n>>> '+' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True,**kwargs)


def prepare(stage):
    import torch
    from alpha2048.value_models import load_value_model
    source = ROOT/'models/latest/best.pt'
    checkpoint = torch.load(source,map_location='cpu',weights_only=True)
    target = stage/'models/latest/best.pt'
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save({k:checkpoint[k] for k in ('model_type','model_config','model')},target)
    original,exported = load_value_model(source),load_value_model(target)
    for name,value in original.state_dict().items():
        assert torch.equal(value,exported.state_dict()[name]),name
    (stage/'model-provenance.json').write_text(json.dumps({
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'inference_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'removed':'optimizer, scheduler, training metrics and metadata; parameters unchanged'},indent=2),encoding='utf-8')


def notices(destination):
    shutil.copytree(ROOT/'packaging/notices',destination,dirs_exist_ok=True)
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get('Name','unknown')
        versions[name] = dist.version
        for item in dist.files or []:
            if not any(token in item.name.lower() for token in ('license','licence','copying','copyright','notice')):
                continue
            source = Path(dist.locate_file(item))
            if not source.is_file() or source.suffix.lower() in ('.py','.pyc','.exe','.dll'):
                continue
            target = destination/name/str(item).replace('../','').replace('..\\','')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    python_license = Path(sys.base_prefix)/'LICENSE.txt'
    if python_license.is_file():
        shutil.copy2(python_license,destination/'Python-LICENSE.txt')
    for source in (Path(sys.base_prefix)/'tcl').rglob('license*'):
        if source.is_file():
            target = destination/'Tcl-Tk'/source.relative_to(Path(sys.base_prefix)/'tcl')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    (destination/'build-dependencies.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')


def cuda_binaries(bundle):
    prefixes = ('torch_cuda','cublas','cudnn','cufft','cusparse','cusolver',
                'cudart','nvrtc','nvjitlink','curand','cupti','c10_cuda')
    return [p for p in bundle.rglob('*') if p.is_file()
            and p.suffix.lower() in ('.dll','.pyd')
            and p.name.lower().startswith(prefixes)]


def finish(output,stage):
    bundle = output/'Alpha2048-CPU'
    for executable in ('2048.exe','ScreenAssistant.exe'):
        report = stage/(executable+'.self-test.json')
        result = json.loads(report.read_text(encoding='utf-8'))
        assert result['ok'] and result['torch'].endswith('+cpu'), 'CPU self-test required'
        assert Path(result['resources']).resolve() == (bundle/'_internal').resolve()
        assert report.stat().st_mtime >= (bundle/executable).stat().st_mtime, 'Executable changed after self-test'
    found = cuda_binaries(bundle)
    assert not found, 'CUDA libraries found: '+', '.join(str(p) for p in found)
    manifest = {str(p.relative_to(bundle)):sha256(p)
                for p in bundle.rglob('*') if p.is_file() and p.name!='SHA256.json'}
    (bundle/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive = Path(shutil.make_archive(str(output/'Alpha2048-CPU-Windows-x64'),'zip',output,bundle.name))
    checksum = sha256(archive)
    archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n',encoding='ascii')
    print(f'\nSUCCESS\nFolder: {bundle}\nZIP: {archive}\nSHA256: {checksum}',flush=True)
    print('Please manually open both EXEs and test a live board before public release.',flush=True)


def build():
    import torch
    assert torch.version.cuda is None,'Refusing to bundle CUDA Torch as a CPU build'
    tag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    stage = ROOT/'build'/f'cpu-{tag}'
    stage.mkdir(parents=True)
    prepare(stage)
    environment = os.environ.copy()
    environment['ALPHA2048_STAGE'] = str(stage)
    environment['PYINSTALLER_CONFIG_DIR'] = str(ROOT/'build/pyinstaller-cache')
    output = ROOT/'dist'/f'cpu-{tag}'
    run([sys.executable,'-m','PyInstaller','--noconfirm','--distpath',output,
         '--workpath',stage/'pyinstaller',ROOT/'packaging/cpu.spec'],env=environment,cwd=ROOT)
    bundle = output/'Alpha2048-CPU'
    shutil.copy2(bundle/'2048.exe',bundle/'ScreenAssistant.exe')
    shutil.copy2(ROOT/'packaging/USER-README.txt',bundle/'README.txt')
    shutil.copy2(stage/'model-provenance.json',bundle/'model-provenance.json')
    notices(bundle/'ThirdPartyNotices')
    # A fresh working directory detects accidental dependencies on the checkout.
    smoke = stage/'unrelated-working-directory'
    smoke.mkdir()
    for executable in ('2048.exe','ScreenAssistant.exe'):
        result = stage/(executable+'.self-test.json')
        run([bundle/executable,'--self-test',result],cwd=smoke,timeout=120)
        assert json.loads(result.read_text(encoding='utf-8'))['ok']
    finish(output,stage)


def main():
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT))
    if len(sys.argv)==3 and sys.argv[1]=='--finish':
        import re
        tag = sys.argv[2]
        if not re.fullmatch(r'cpu-\d{8}-\d{6}',tag):
            raise ValueError('Expected cpu-YYYYMMDD-HHMMSS')
        finish(ROOT/'dist'/tag,ROOT/'build'/tag)
        return
    if '--build' in sys.argv:
        build()
        return
    if '--check' in sys.argv:
        for name in ('models/latest/best.pt','assets/ocr/en_PP-OCRv4_rec.onnx','assets/ocr/dict.txt',
                     'packaging/notices/OCR-conversion-LICENCE.txt','packaging/notices/PaddleOCR-LICENSE.txt'):
            assert (ROOT/name).is_file(),name
        print('Build inputs present. No installation or build started.')
        return
    if not (ENV/'Scripts/python.exe').is_file():
        run([sys.executable,'-m','venv',ENV])
    python = ENV/'Scripts/python.exe'
    run([python,'-m','pip','install','torch==2.14.0','--index-url','https://download.pytorch.org/whl/cpu'])
    run([python,'-m','pip','install',*PINS])
    run([python,'-m','pip','check'])
    run([python,__file__,'--build'])


if __name__=='__main__':
    main()
