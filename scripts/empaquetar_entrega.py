"""Empaqueta datos, notebooks y funciones en un único .py reproducible."""
from pathlib import Path
import json, base64, zlib, hashlib, importlib.metadata
import nbformat
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'entrega';OUT.mkdir(exist_ok=True)
files={}
for p in (ROOT/'data').glob('*.csv'):files[str(p.relative_to(ROOT))]=base64.b64encode(p.read_bytes()).decode()
for n in range(2,13):
    path=ROOT/f'puntos/punto{n}/punto{n}.ipynb';nb=nbformat.read(path,as_version=4)
    for c in nb.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    files[str(path.relative_to(ROOT))]=base64.b64encode(nbformat.writes(nb).encode()).decode()
paths=['puntos/punto1/exposicion del problema.txt','scripts/modelos_puntos56.py','scripts/modelos_resto.py','scripts/crear_informe.py']
for name in paths:files[name]=base64.b64encode((ROOT/name).read_bytes()).decode()
versions=[]
for name in ['pandas','numpy','matplotlib','seaborn','statsmodels','scipy','arch','nbformat','nbclient','ipykernel','reportlab']:
    versions.append(f'{name}=={importlib.metadata.version(name)}')
(OUT/'versiones_entorno.txt').write_text('\n'.join(versions)+'\n')
files['requirements_reproduccion.txt']=base64.b64encode(('\n'.join(versions)+'\n').encode()).decode()
payload=base64.b64encode(zlib.compress(json.dumps(files).encode(),9)).decode()
script='''#!/usr/bin/env python3
"""TP1 — Series temporales de Rosario.

Autores: Tomas del Bo, Matias Villanueva y Juan Ignacio Paberolis.

Reproduce los puntos 2–12 y el informe; incluye los cuatro CSV utilizados.
Uso:
    python tp1_reproducible.py --solo-extraer --output revision_tp1
    python -m pip install -r revision_tp1/requirements_reproduccion.txt
    python tp1_reproducible.py --output reproduccion_tp1

La carpeta de destino debe estar vacía. La extracción es local: no descarga
ni envía datos. El código de cada punto queda visible en los notebooks y
los módulos Python extraídos. El contenido comprimido evita necesitar
archivos adjuntos adicionales para reconstruir exactamente los datos.
"""
from pathlib import Path
import argparse, base64, json, os, subprocess, sys, zlib

# Paquete local de archivos: JSON comprimido, sin pickle ni código ejecutado al extraer.
PAYLOAD = """__PAYLOAD__"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('reproduccion_tp1'))
    parser.add_argument('--solo-extraer',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('La carpeta de destino debe estar vacía; elija una carpeta nueva.')
    root.mkdir(parents=True,exist_ok=True)
    files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
    for name,content in files.items():
        target=(root/name).resolve()
        if not target.is_relative_to(root):raise ValueError('Ruta fuera del destino')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(base64.b64decode(content))
    print(f'Archivos extraídos en {root}',flush=True)
    if args.solo_extraer:return
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        parser.error(f'Instale las dependencias: python -m pip install -r {root / "requirements_reproduccion.txt"}')
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='1'
    for n in range(2,13):
        path=root/f'puntos/punto{n}/punto{n}.ipynb'
        nb=nbformat.read(path,as_version=4)
        print(f'Ejecutando punto {n}...',flush=True)
        NotebookClient(nb,timeout=900,resources={'metadata':{'path':str(path.parent)}}).execute()
        nbformat.write(nb,path)
    subprocess.run([sys.executable,str(root/'scripts/crear_informe.py')],cwd=root,check=True)
    print(f'Finalizado. Informe: {root / "entrega/TP1_informe.pdf"}',flush=True)

if __name__=='__main__':main()
'''.replace('__PAYLOAD__',payload)
(OUT/'tp1_reproducible.py').write_text(script)
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.csv')}
(OUT/'huellas_datos.json').write_text(json.dumps(manifest,indent=2))
print('Archivo reproducible:',len(script),'bytes')
