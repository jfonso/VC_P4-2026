#!/usr/bin/env python3

import argparse
import contextlib
import csv
import glob
import inspect
import io
import os
import re
import shutil
import statistics
import tempfile
import time

import cv2

# def preprocesar(img_bgr):
#     resized = cv2.resize(img_bgr, (320, int(img_bgr.shape[0] / img_bgr.shape[1] * 320)))
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     _, img_thres = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
#     return img_thres

def preprocesar(img_bgr):
    resized = cv2.resize(img_bgr, (320, int(img_bgr.shape[0] / img_bgr.shape[1] * 320)),
                         interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Polaridad: texto negro sobre fondo blanco (se mira solo el centro)
    H, W = b.shape
    if (b[H//4:3*H//4, W//8:7*W//8] == 0).mean() > 0.5:
        b = 255 - b
    return cv2.copyMakeBorder(b, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)


def postprocesar(texto):
    return re.sub(r'[^0-9A-Z]', '', texto.upper().replace(' ', ''))


def normalizar_referencia(texto):
    return postprocesar(texto)

def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def percentil(valores, p):
    v = sorted(valores)
    if not v:
        return float('nan')
    k = (len(v) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

class OCRTesseract:
    nombre = 'tesseract'

    def __init__(self, cmd):
        import pytesseract
        self.pt = pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd
        self.config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' # '--psm 11'

    def sincronizar(self):
        pass

    def inferir(self, img):
        return self.pt.image_to_string(img, config=self.config)


class OCRUnlimited:
    nombre = 'unlimited-ocr'

    # Etiquetas especiales y bloques de coordenadas que el modelo puede emitir (formato DeepSeek-OCR).
    # Hay que quitarlos ANTES del postprocesado común; si no, los números de las cajas
    # se colarían en la matrícula.
    _RE_DET = re.compile(r'<\|(det|ref)\|>.*?<\|/\1\|>', re.S)
    _RE_TAG = re.compile(r'<\|[^|]*\|>|<PAGE>|</?[a-zA-Z][^>]*>')

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('Unlimited-OCR necesita una GPU NVIDIA con CUDA.')
        self.torch = torch
        self.prompt = '<image>document parsing.'
        self.dtype = getattr(torch, 'bfloat16')
        modelo = 'baidu/Unlimited-OCR'
        self.tok = AutoTokenizer.from_pretrained(modelo, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(modelo, trust_remote_code=True, use_safetensors=True,
                                               torch_dtype=self.dtype).eval().cuda()
        generate_original = self.model.generate
        def generate_limitado(*args, **kwargs):
            kwargs['max_new_tokens'] = 128
            kwargs.pop('max_length', None)
            return generate_original(*args, **kwargs)
        self.model.generate = generate_limitado
        cfg = dict(base_size=1024, image_size=1024, crop_mode=False)
        cfg.update(no_repeat_ngram_size=35, ngram_window=128)
        self.cfg = cfg
        self.params_infer = set(inspect.signature(self.model.infer).parameters)
        self.tmp = tempfile.mkdtemp(prefix='uocr_')
        self.devuelve_texto = None

    def sincronizar(self):
        self.torch.cuda.synchronize()

    def _leer_resultado_guardado(self, carpeta):
        ficheros = [f for ext in ('mmd', 'md', 'txt') for f in glob.glob(os.path.join(carpeta, '**', f'*.{ext}'), recursive=True)]
        if not ficheros:
            return ''
        with open(max(ficheros, key=os.path.getmtime), encoding='utf-8') as fh:
            return fh.read()

    def inferir(self, img):
        ruta = os.path.join(self.tmp, 'entrada.png')
        cv2.imwrite(ruta, img)
        salida = tempfile.mkdtemp(dir=self.tmp)
        guardar = self.devuelve_texto is not True
        kwargs = dict(prompt=self.prompt, image_file=ruta, output_path=salida, save_results=guardar, **self.cfg)
        if 'eval_mode' in self.params_infer:
            kwargs['eval_mode'] = True
        kwargs = {k: v for k, v in kwargs.items() if k in self.params_infer or 'kwargs' in self.params_infer}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.torch.inference_mode():
                res = self.model.infer(self.tok, **kwargs)
        if isinstance(res, (tuple, list)):
            res = res[0]
        if isinstance(res, str) and res.strip():
            self.devuelve_texto = True
            texto = res
        else:
            self.devuelve_texto = False
            texto = self._leer_resultado_guardado(salida)
        shutil.rmtree(salida, ignore_errors=True)
        return texto

    @classmethod
    def limpiar_formato(cls, texto):
        texto = cls._RE_DET.sub(' ', texto)
        return cls._RE_TAG.sub(' ', texto)


def cargar_dataset(carpeta, limite=None):
    with open(os.path.join(carpeta, 'labels.csv'), newline='', encoding='utf-8') as fh:
        filas = list(csv.DictReader(fh))
    if limite:
        filas = filas[:limite]
    datos = []
    for f in filas:
        img = cv2.imread(os.path.join(carpeta, 'images', f['file']))
        if img is None:
            print(f"AVISO: no se pudo leer {f['file']}, se omite")
            continue
        t0 = time.perf_counter()
        pre = preprocesar(img)
        t_pre = (time.perf_counter() - t0) * 1000
        datos.append({**f, 'img_pre': pre, 't_preproceso_ms': t_pre})
    return datos


def evaluar(ocr, datos, warmup):
    print(f'\n== {ocr.nombre}: warm-up con {warmup} imágenes')
    for d in datos[:warmup]:
        ocr.inferir(d['img_pre'])
    ocr.sincronizar()

    filas = []
    for n, d in enumerate(datos, 1):
        ocr.sincronizar()
        t0 = time.perf_counter()
        crudo = ocr.inferir(d['img_pre'])
        ocr.sincronizar()
        tiempo = ((time.perf_counter() - t0) * 1000)
        limpio = OCRUnlimited.limpiar_formato(crudo) if isinstance(ocr, OCRUnlimited) else crudo
        pred = postprocesar(limpio)
        ref = normalizar_referencia(d['text'])
        dist = levenshtein(pred, ref)
        filas.append({
            'modelo': ocr.nombre, 'file': d['file'], 'referencia': ref, 'prediccion': pred,
            'salida_cruda': crudo.replace('\n', '\\n').strip(),
            'acierto_exacto': int(pred == ref), 'distancia_edicion': dist,
            'cer': round(dist / max(len(ref), 1), 4), 'salida_vacia': int(pred == ''),
            't_inferencia_ms': round(tiempo, 3),
            't_preproceso_ms': round(d['t_preproceso_ms'], 3),
            'country': d.get('country', ''), 'lines': d.get('lines', ''), 'quality': d.get('quality', ''),
        })
        print(f"[{n:3d}/{len(datos)}] {d['file']}  ref={ref:<12} pred={pred:<14} "
              f"{'OK ' if pred == ref else 'MAL'} {filas[-1]['t_inferencia_ms']:8.1f} ms")
    return filas


def resumir(filas, modelo, t_carga, mem_gpu_mb=None, subconjunto='todas'):
    t = [f['t_inferencia_ms'] for f in filas]
    n = len(filas)
    acc = sum(f['acierto_exacto'] for f in filas) / n
    total_ref = sum(len(f['referencia']) for f in filas)
    return {
        'modelo': modelo, 'subconjunto': subconjunto, 'n_imagenes': n,
        'exact_match': round(acc, 4),
        'ic95_exact_match': round(1.96 * (acc * (1 - acc) / n) ** 0.5, 4),
        'cer_global': round(sum(f['distancia_edicion'] for f in filas) / max(total_ref, 1), 4),
        'cer_medio': round(statistics.mean(f['cer'] for f in filas), 4),
        'salidas_vacias': round(sum(f['salida_vacia'] for f in filas) / n, 4),
        't_media_ms': round(statistics.mean(t), 2), 't_mediana_ms': round(statistics.median(t), 2),
        't_p95_ms': round(percentil(t, 95), 2), 't_std_ms': round(statistics.pstdev(t), 2),
        't_min_ms': round(min(t), 2), 't_max_ms': round(max(t), 2),
        'imagenes_por_s': round(1000 / statistics.mean(t), 2),
        't_carga_modelo_s': round(t_carga, 2),
        'mem_gpu_pico_mb': mem_gpu_mb if mem_gpu_mb is not None else '',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='plates_ocr', help='carpeta con images/ y labels.csv')
    ap.add_argument('--salida', default='resultados')
    ap.add_argument('--warmup', type=int, default=5, help='inferencias previas no medidas')
    ap.add_argument('--limite', type=int, help='usar solo las N primeras imágenes (pruebas)')
    ap.add_argument('--tesseract-cmd', help='ruta al ejecutable de Tesseract si no está en el PATH')
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)

    datos = cargar_dataset(args.dataset, args.limite)
    print(f'{len(datos)} imágenes cargadas y preprocesadas')

    todas, resumen = [], []
    for nombre in ['tesseract', 'unlimited']:
        t0 = time.perf_counter()
        if nombre == 'tesseract':
            ocr = OCRTesseract(args.tesseract_cmd)
        else:
            ocr = OCRUnlimited()
            ocr.torch.cuda.reset_peak_memory_stats()
        t_carga = time.perf_counter() - t0

        filas = evaluar(ocr, datos, args.warmup)
        mem = None
        if isinstance(ocr, OCRUnlimited):
            mem = round(ocr.torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        todas += filas
        resumen.append(resumir(filas, ocr.nombre, t_carga, mem))
        for q in ('alta', 'media'):
            sub = [f for f in filas if f['quality'] == q]
            if sub:
                resumen.append(resumir(sub, ocr.nombre, t_carga, mem, f'quality={q}'))
        del ocr

    with open(os.path.join(args.salida, 'resultados.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)
    with open(os.path.join(args.salida, 'resumen.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)

    print('\n== Resumen')
    for r in resumen:
        print(f"{r['modelo']:<14} {r['subconjunto']:<14} n={r['n_imagenes']:<4} exact={r['exact_match']:.3f}"
              f"±{r['ic95_exact_match']:.3f}  CER={r['cer_global']:.3f}  "
              f"t_media={r['t_media_ms']:.1f} ms  p95={r['t_p95_ms']:.1f} ms")
    print(f"\nFicheros guardados en {os.path.abspath(args.salida)}")


if __name__ == '__main__':
    main()
