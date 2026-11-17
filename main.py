"""
Pizarra virtual controlada con senas de la mano.

FASE 1 - Dibujar
  * Solo el INDICE levantado ............ dibuja (o borra si el borrador esta activo)
  * UNA palma abierta ................... abre la paleta de colores (oculta por defecto).
                                          Con el dedo indice pasa sobre un color, un
                                          tamano de pincel o "Cerrar" para elegirlo.
  * DOS palmas abiertas ................. activa el borrador: luego, con el indice,
                                          lo que toques se va borrando. Para volver al
                                          pincel elige un color en la paleta.

FASE 2 - Senas
  * Like (pulgar arriba), Dislike (pulgar abajo)
  * Corazon con las dos manos (indices juntos arriba, pulgares juntos abajo)
  * Senas obscenas (dedo medio, "higa") -> se CENSURAN con pixelado

Teclas: q/ESC salir | c limpiar | s guardar dibujo | p abrir/cerrar paleta
        d dibujo on/off | g senas on/off | l puntos de la mano | i depuracion
"""
import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from gestos import OBSCENOS, Mano, es_corazon

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(BASE, "models", "hand_landmarker.task")
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
CARPETA_CAPTURAS = os.path.join(BASE, "capturas")
VENTANA = "Pizarra de gestos"

ANCHO, ALTO = 1280, 720

COLORES = [
    ("Rojo", (0, 0, 255)),
    ("Naranja", (0, 140, 255)),
    ("Amarillo", (0, 230, 255)),
    ("Verde", (0, 200, 0)),
    ("Azul", (255, 90, 0)),
    ("Morado", (200, 0, 200)),
    ("Rosa", (180, 105, 255)),
    ("Blanco", (255, 255, 255)),
]
TAMANOS = [("Chico", 5), ("Mediano", 14), ("Grande", 28)]
GROSOR_BORRADOR = 70

TIEMPO_SELECCION = 0.35   # segundos que hay que mantener el dedo sobre un boton
TIEMPO_LIMPIAR = 0.9      # "Limpiar" pide mas tiempo para evitar accidentes
SUAVIZADO = 0.5           # 0 = sin suavizado, cercano a 1 = muy suave (mas lento)

# Estabilidad de las senas (en cuadros)
VENTANA_HIST = 8
MIN_CUADROS_SENA = 5      # like / dislike / corazon
MIN_CUADROS_PALMA = 6     # para abrir paleta o activar borrador
MIN_CUADROS_CENSURA = 2   # la censura reacciona rapido a proposito
MANTENER_CENSURA = 0.7    # segundos que sigue la censura tras dejar de verse
AVISO_DURACION = 1.5      # segundos que se muestra un aviso ("Borrador activado")

CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


@dataclass
class Boton:
    x1: int
    y1: int
    x2: int
    y2: int
    tipo: str      # "color" | "tamano" | "limpiar" | "cerrar"
    valor: object
    etiqueta: str

    def contiene(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Paleta:
    x1: int
    y1: int
    x2: int
    y2: int
    botones: list


# ---------------------------------------------------------------------------
# Preparacion
# ---------------------------------------------------------------------------
def asegurar_modelo():
    if os.path.exists(MODELO):
        return
    os.makedirs(os.path.dirname(MODELO), exist_ok=True)
    print("Descargando modelo de manos de MediaPipe...")
    urllib.request.urlretrieve(URL_MODELO, MODELO)


def abrir_camara():
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
            return cap
        cap.release()
    raise RuntimeError("No se pudo abrir la camara. Revisa que no la use otra aplicacion.")


def crear_paleta(ancho, alto):
    """Panel centrado: fila de colores, fila de tamanos + Limpiar, y boton Cerrar."""
    pw, ph = int(ancho * 0.8), int(alto * 0.55)
    x1, y1 = (ancho - pw) // 2, (alto - ph) // 2
    x2, y2 = x1 + pw, y1 + ph
    margen, sep = 20, 12
    botones = []

    # Boton cerrar (esquina superior derecha)
    cw, ch = 130, 50
    botones.append(Boton(x2 - margen - cw, y1 + margen, x2 - margen, y1 + margen + ch,
                         "cerrar", None, "Cerrar X"))

    # Fila de colores
    fila1_y1 = y1 + margen + ch + 20
    fila_h = (y2 - margen - fila1_y1 - 20) // 2
    n = len(COLORES)
    bw = (pw - 2 * margen - sep * (n - 1)) // n
    for i, (nombre, color) in enumerate(COLORES):
        bx = x1 + margen + i * (bw + sep)
        botones.append(Boton(bx, fila1_y1, bx + bw, fila1_y1 + fila_h, "color", color, nombre))

    # Fila de tamanos + limpiar
    fila2_y1 = fila1_y1 + fila_h + 20
    n = len(TAMANOS) + 1
    bw = (pw - 2 * margen - sep * (n - 1)) // n
    for i, (nombre, grosor) in enumerate(TAMANOS):
        bx = x1 + margen + i * (bw + sep)
        botones.append(Boton(bx, fila2_y1, bx + bw, fila2_y1 + fila_h, "tamano", grosor, nombre))
    bx = x1 + margen + len(TAMANOS) * (bw + sep)
    botones.append(Boton(bx, fila2_y1, bx + bw, fila2_y1 + fila_h, "limpiar", None, "Limpiar todo"))
    return Paleta(x1, y1, x2, y2, botones)


# ---------------------------------------------------------------------------
# Dibujo de interfaz
# ---------------------------------------------------------------------------
def texto(img, txt, pos, escala=0.6, color=(255, 255, 255), grosor=1, fondo=None):
    (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    x, y = pos
    if fondo is not None:
        cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 6, y + base + 4), fondo, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def texto_centrado(img, txt, cx, cy, escala=0.6, color=(255, 255, 255), grosor=1):
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    cv2.putText(img, txt, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                escala, color, grosor, cv2.LINE_AA)


def dibujar_paleta(img, paleta, estado, hover, progreso):
    capa = img.copy()
    cv2.rectangle(capa, (paleta.x1, paleta.y1), (paleta.x2, paleta.y2), (35, 35, 35), -1)
    cv2.addWeighted(capa, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (paleta.x1, paleta.y1), (paleta.x2, paleta.y2), (200, 200, 200), 2)
    texto(img, "Paleta", (paleta.x1 + 20, paleta.y1 + 50), 0.9, (255, 255, 255), 2)

    for b in paleta.botones:
        activo = ((b.tipo == "color" and not estado["borrador"] and b.valor == estado["color"]) or
                  (b.tipo == "tamano" and b.valor == estado["grosor"]))
        cx, cy = (b.x1 + b.x2) // 2, (b.y1 + b.y2) // 2
        if b.tipo == "color":
            cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), b.valor, -1)
        elif b.tipo == "cerrar":
            cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (40, 40, 190), -1)
            texto_centrado(img, b.etiqueta, cx, cy, 0.7, (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (70, 70, 70), -1)
            if b.tipo == "tamano":
                cv2.circle(img, (cx, cy - 12), max(2, b.valor // 2), estado["color"], -1, cv2.LINE_AA)
                texto_centrado(img, b.etiqueta, cx, b.y2 - 18, 0.55)
            else:
                texto_centrado(img, b.etiqueta, cx, cy, 0.6)

        borde = (255, 255, 255) if activo else (120, 120, 120)
        cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), borde, 5 if activo else 1)
        if b is hover and progreso > 0:
            ancho = int((b.x2 - b.x1) * min(progreso, 1.0))
            cv2.rectangle(img, (b.x1, b.y2 + 4), (b.x1 + ancho, b.y2 + 10), (0, 255, 255), -1)


def dibujar_corazon(img, centro, tam, color=(147, 20, 255)):
    cx, cy = centro
    r = tam // 2
    cv2.circle(img, (cx - r // 2 - 2, cy), r // 2 + 2, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx + r // 2 + 2, cy), r // 2 + 2, color, -1, cv2.LINE_AA)
    pts = np.array([[cx - r - 2, cy + 4], [cx + r + 2, cy + 4], [cx, cy + r + r // 2 + 4]], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def dibujar_pulgar(img, centro, arriba, color):
    """Icono sencillo de pulgar (rectangulo del puno + pulgar)."""
    cx, cy = centro
    cv2.rectangle(img, (cx - 22, cy - 15), (cx + 22, cy + 25), color, -1)
    if arriba:
        cv2.rectangle(img, (cx - 22, cy - 50), (cx - 6, cy - 12), color, -1)
    else:
        cv2.rectangle(img, (cx - 22, cy + 22), (cx - 6, cy + 60), color, -1)


def mostrar_sena(img, sena):
    w = img.shape[1]
    estilos = {
        "like": ("LIKE", (0, 170, 0)),
        "dislike": ("DISLIKE", (0, 0, 200)),
        "corazon": ("CORAZON", (147, 20, 255)),
    }
    etiqueta, color = estilos[sena]
    x, y = w - 260, 20
    cv2.rectangle(img, (x, y), (w - 20, y + 120), (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (w - 20, y + 120), color, 3)
    centro = (x + 55, y + 55)
    if sena == "corazon":
        dibujar_corazon(img, centro, 60, color)
    else:
        dibujar_pulgar(img, centro, sena == "like", color)
    texto(img, etiqueta, (x + 100, y + 70), 0.8, (255, 255, 255), 2)


def pixelar(img, x1, y1, x2, y2, bloques=10):
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    roi = img[y1:y2, x1:x2]
    chico = cv2.resize(roi, (bloques, max(1, int(bloques * (y2 - y1) / (x2 - x1)))),
                       interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = cv2.resize(chico, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    texto(img, "CENSURADO", (x1 + 4, max(20, y1 - 8)), 0.7, (255, 255, 255), 2, fondo=(0, 0, 200))


def dibujar_esqueleto(img, mano):
    pts = mano.img.astype(int)
    for a, b in CONEXIONES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 3, (0, 0, 255), -1)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    asegurar_modelo()
    opciones = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = abrir_camara()
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)

    estado = {"color": COLORES[0][1], "grosor": TAMANOS[1][1], "borrador": False}
    # Los puntos de la mano estan ocultos por defecto (tecla "l" para verlos)
    opciones_vista = {"dibujo": True, "senas": True, "esqueleto": False, "debug": False}
    lienzo, paleta = None, None
    paleta_abierta = False
    esperar_sin_palma = False   # tras cerrar la paleta, hay que bajar la palma para reabrirla
    esperar_sin_dos_palmas = False
    punto_prev, punta_suave = None, None
    hover, hover_inicio = None, 0.0
    historial = deque(maxlen=VENTANA_HIST)
    censura_hasta = 0.0
    aviso, aviso_hasta = "", 0.0
    ts_prev, t_fps, fps = 0, time.time(), 0.0

    with vision.HandLandmarker.create_from_options(opciones) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer la camara.")
                break
            frame = cv2.flip(frame, 1)  # efecto espejo
            h, w = frame.shape[:2]
            if lienzo is None or lienzo.shape != frame.shape:
                lienzo = np.zeros_like(frame)
                paleta = crear_paleta(w, h)

            # ---- deteccion de manos
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(ts_prev + 1, int(time.monotonic() * 1000))
            ts_prev = ts
            resultado = detector.detect_for_video(imagen_mp, ts)

            manos = []
            for lms, wlms in zip(resultado.hand_landmarks, resultado.hand_world_landmarks):
                pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
                p3 = np.array([[p.x, p.y, p.z] for p in wlms], dtype=np.float32)
                manos.append(Mano(pts, p3))
            gestos = [m.clasificar() for m in manos]

            # ---- senas del cuadro actual
            senas = set()
            n_palmas = gestos.count("palma")
            if n_palmas >= 2:
                senas.add("dos_palmas")
            elif n_palmas == 1:
                senas.add("palma")
            if opciones_vista["senas"]:
                senas.update(g for g in gestos if g in OBSCENOS | {"like", "dislike"})
                if len(manos) == 2 and es_corazon(manos[0], manos[1]):
                    senas.add("corazon")
            historial.append(senas)

            def veces(nombre):
                return sum(nombre in s for s in historial)

            ahora = time.time()
            if any(veces(o) >= MIN_CUADROS_CENSURA for o in OBSCENOS):
                censura_hasta = ahora + MANTENER_CENSURA
            censurando = ahora < censura_hasta
            sena_estable = next((s for s in ("corazon", "like", "dislike")
                                 if veces(s) >= MIN_CUADROS_SENA), None)

            # ---- dos palmas abiertas -> borrador
            if veces("dos_palmas") == 0:
                esperar_sin_dos_palmas = False
            if (veces("dos_palmas") >= MIN_CUADROS_PALMA and not esperar_sin_dos_palmas
                    and not censurando):
                esperar_sin_dos_palmas = True
                if not estado["borrador"]:
                    estado["borrador"] = True
                    aviso, aviso_hasta = "Borrador activado", ahora + AVISO_DURACION

            # ---- una palma abierta -> abrir paleta
            # (no se abre si en los ultimos cuadros se vieron dos palmas)
            if veces("palma") == 0 and veces("dos_palmas") == 0:
                esperar_sin_palma = False
            if (not paleta_abierta and not esperar_sin_palma and not censurando
                    and veces("palma") >= MIN_CUADROS_PALMA and veces("dos_palmas") == 0):
                paleta_abierta = True

            # ---- puntero (indice levantado)
            modo = "pausa"
            progreso = 0.0
            hover_actual = None
            principal = None
            if opciones_vista["dibujo"] and not censurando and sena_estable is None:
                for m, g in zip(manos, gestos):
                    if g in ("dibujar", "seleccionar"):
                        principal = m
                        break

            if principal is not None:
                punta = principal.punta_indice()
                if punta_suave is None:
                    punta_suave = punta.copy()
                else:
                    punta_suave = SUAVIZADO * punta_suave + (1 - SUAVIZADO) * punta
                x, y = int(punta_suave[0]), int(punta_suave[1])

                if paleta_abierta:
                    # Seleccion en la paleta: mantener el dedo sobre un boton
                    modo = "paleta"
                    punto_prev = None
                    hover_actual = next((b for b in paleta.botones if b.contiene(x, y)), None)
                    if hover_actual is not None:
                        if hover_actual is not hover:
                            hover, hover_inicio = hover_actual, ahora
                        necesario = TIEMPO_LIMPIAR if hover.tipo == "limpiar" else TIEMPO_SELECCION
                        progreso = (ahora - hover_inicio) / necesario
                        if progreso >= 1.0:
                            if hover.tipo == "color":
                                estado["color"], estado["borrador"] = hover.valor, False
                            elif hover.tipo == "tamano":
                                estado["grosor"] = hover.valor
                                estado["borrador"] = False
                            elif hover.tipo == "limpiar":
                                lienzo[:] = 0
                            elif hover.tipo == "cerrar":
                                paleta_abierta = False
                                esperar_sin_palma = True
                            hover_inicio = ahora + 10  # evita repetir la accion
                else:
                    # Dibujar o borrar con el indice
                    modo = "borrar" if estado["borrador"] else "dibujar"
                    color = (0, 0, 0) if estado["borrador"] else estado["color"]
                    grosor = GROSOR_BORRADOR if estado["borrador"] else estado["grosor"]
                    if punto_prev is not None:
                        cv2.line(lienzo, punto_prev, (x, y), color, grosor, cv2.LINE_AA)
                    else:
                        cv2.circle(lienzo, (x, y), max(1, grosor // 2), color, -1, cv2.LINE_AA)
                    punto_prev = (x, y)
            else:
                punto_prev, punta_suave = None, None
            if hover_actual is None:
                hover = None

            # ---- composicion: camara + dibujo
            salida = frame.copy()
            gris = cv2.cvtColor(lienzo, cv2.COLOR_BGR2GRAY)
            mascara = gris > 10
            salida[mascara] = lienzo[mascara]

            # ---- censura (y puntos de la mano solo si se activan con "l")
            for m, g in zip(manos, gestos):
                obscena = g in OBSCENOS and opciones_vista["senas"]
                if censurando or obscena:
                    pixelar(salida, *m.caja(0.35, (w, h)))
                elif opciones_vista["esqueleto"]:
                    dibujar_esqueleto(salida, m)

            if paleta_abierta:
                dibujar_paleta(salida, paleta, estado, hover, progreso)

            if opciones_vista["debug"]:
                for i, (m, g) in enumerate(zip(manos, gestos)):
                    m._es_higa()  # rellena valores de depuracion
                    lineas = [f"Mano {i + 1}: {g}",
                              "  " + " ".join(f"{k[:3]}:{v}" for k, v in m.estado.items()),
                              f"  pulgar_ext:{m.pulgar_ext}  " +
                              " ".join(f"{k[:3]}:{v:.0f}" for k, v in m.angulos.items()),
                              f"  higa alt:{m.debug['higa_alt']:.2f} dist:{m.debug['higa_dist']:.2f}"
                              f" entre:{m.debug['higa_entre']}"]
                    for j, linea in enumerate(lineas):
                        texto(salida, linea, (10, 30 + (i * 4 + j) * 22), 0.5,
                              (0, 255, 255), 1, fondo=(0, 0, 0))

            # ---- cursor
            if principal is not None:
                x, y = int(punta_suave[0]), int(punta_suave[1])
                if modo == "paleta":
                    cv2.circle(salida, (x, y), 14, (255, 255, 255), 2)
                    cv2.circle(salida, (x, y), 3, (255, 255, 255), -1)
                elif modo == "borrar":
                    cv2.circle(salida, (x, y), GROSOR_BORRADOR // 2, (230, 230, 230), 2)
                else:
                    cv2.circle(salida, (x, y), max(3, estado["grosor"] // 2), estado["color"], 2)

            # ---- avisos y senas
            if sena_estable and not censurando:
                mostrar_sena(salida, sena_estable)
            if ahora < aviso_hasta:
                texto_centrado(salida, aviso, w // 2, 40, 1.0, (255, 255, 255), 2)

            # ---- barra de estado
            ahora_fps = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, ahora_fps - t_fps))
            t_fps = ahora_fps
            nombres_modo = {"dibujar": "DIBUJANDO", "borrar": "BORRANDO",
                            "paleta": "PALETA", "pausa": "PAUSA"}
            herramienta = "Borrador" if estado["borrador"] else f"Pincel {estado['grosor']}px"
            texto(salida, f"{nombres_modo[modo]} | {herramienta} | FPS {fps:.0f}",
                  (10, h - 40), 0.6, (255, 255, 255), 1, fondo=(0, 0, 0))
            if not estado["borrador"]:
                cv2.rectangle(salida, (w - 50, h - 58), (w - 15, h - 23), estado["color"], -1)
                cv2.rectangle(salida, (w - 50, h - 58), (w - 15, h - 23), (255, 255, 255), 1)
            texto(salida, "1 palma: paleta | 2 palmas: borrador | indice: dibujar | "
                          "q salir  c limpiar  s guardar",
                  (10, h - 12), 0.5, (220, 220, 220), 1, fondo=(0, 0, 0))

            cv2.imshow(VENTANA, salida)

            # ---- teclado
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("c"):
                lienzo[:] = 0
            elif tecla == ord("p"):
                paleta_abierta = not paleta_abierta
                esperar_sin_palma = not paleta_abierta
            elif tecla == ord("s"):
                os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
                nombre = time.strftime("dibujo_%Y%m%d_%H%M%S.png")
                cv2.imwrite(os.path.join(CARPETA_CAPTURAS, nombre), salida)
                print("Guardado:", nombre)
            elif tecla in (ord("d"), ord("g"), ord("l"), ord("i")):
                clave = {"d": "dibujo", "g": "senas", "l": "esqueleto", "i": "debug"}[chr(tecla)]
                opciones_vista[clave] = not opciones_vista[clave]
            if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
