"""
Clasificacion de gestos de mano a partir de los 21 puntos (landmarks) de MediaPipe.

Se usan dos juegos de coordenadas por mano:
  - img : (21, 2) en pixeles de la imagen  -> posiciones, direcciones arriba/abajo, cajas
  - w3  : (21, 3) "world landmarks" en metros -> angulos de los dedos (no dependen de la
          rotacion de la mano frente a la camara, por eso son mas confiables)

Indices de MediaPipe:
  0 muneca
  1-4   pulgar  (CMC, MCP, IP, TIP)
  5-8   indice  (MCP, PIP, DIP, TIP)
  9-12  medio
  13-16 anular
  17-20 menique
"""
import numpy as np

# ---------------------------------------------------------------------------
# Umbrales (ajustalos si la deteccion no te funciona bien; usa la tecla "i"
# en el programa para ver los valores en vivo)
# ---------------------------------------------------------------------------
ANG_EXTENDIDO = 150      # angulo en la articulacion PIP para considerar el dedo estirado
ANG_DOBLADO = 120        # por debajo de esto el dedo esta doblado
ANG_PULGAR = 145         # angulos del pulgar (MCP e IP) para considerarlo estirado
PULGAR_DIR = 0.65        # componente vertical minima del pulgar para like / dislike

# "Higa" (pulgar asomando entre indice y medio con el puno cerrado)
HIGA_ALTURA = 0.80       # altura minima del pulgar relativa a los nudillos PIP
HIGA_DISTANCIA = 0.55    # distancia maxima del pulgar al punto medio entre PIP indice/medio
HIGA_MARGEN_LATERAL = 0.20

# Corazon con dos manos
CORAZON_PUNTAS = 0.55    # distancia maxima entre puntas de indices / entre pulgares
CORAZON_ALTO = 0.45      # los pulgares deben estar por debajo de los indices

EXT, DOB, MEDIO = "ext", "dob", "medio"

DEDOS = {
    "indice": (5, 6, 7, 8),
    "medio": (9, 10, 11, 12),
    "anular": (13, 14, 15, 16),
    "menique": (17, 18, 19, 20),
}

# Gestos que se consideran obscenos y se censuran
OBSCENOS = {"dedo_medio", "higa"}


def _angulo(a, b, c):
    """Angulo en grados en el vertice b."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class Mano:
    def __init__(self, pts_img, pts_3d):
        self.img = pts_img
        self.w3 = pts_3d
        self.escala_img = float(np.linalg.norm(pts_img[0] - pts_img[9])) + 1e-6
        self.escala_3d = float(np.linalg.norm(pts_3d[0] - pts_3d[9])) + 1e-6
        self.angulos = {}
        self.estado = {n: self._estado_dedo(n, *idx) for n, idx in DEDOS.items()}
        self.pulgar_ext = self._pulgar_extendido()
        self.debug = {}

    # --------------------------------------------------------------- dedos
    def _estado_dedo(self, nombre, mcp, pip, dip, tip):
        p = self.w3
        a_pip = _angulo(p[mcp], p[pip], p[dip])
        a_dip = _angulo(p[pip], p[dip], p[tip])
        self.angulos[nombre] = a_pip
        lejos = np.linalg.norm(p[tip] - p[0]) > np.linalg.norm(p[pip] - p[0])
        if a_pip > ANG_EXTENDIDO and a_dip > 130 and lejos:
            return EXT
        if a_pip < ANG_DOBLADO or not lejos:
            return DOB
        return MEDIO

    def _pulgar_extendido(self):
        p = self.w3
        a_mcp = _angulo(p[1], p[2], p[3])
        a_ip = _angulo(p[2], p[3], p[4])
        self.angulos["pulgar"] = min(a_mcp, a_ip)
        separado = np.linalg.norm(p[4] - p[5]) / self.escala_3d > 0.55
        return a_mcp > ANG_PULGAR and a_ip > ANG_PULGAR and separado

    def todos(self, estado, *nombres):
        return all(self.estado[n] == estado for n in nombres)

    # --------------------------------------------------------------- utilidades
    def punta_indice(self):
        return self.img[8]

    def caja(self, margen=0.25, limites=None):
        x1, y1 = self.img.min(axis=0)
        x2, y2 = self.img.max(axis=0)
        mx, my = (x2 - x1) * margen, (y2 - y1) * margen
        x1, y1, x2, y2 = int(x1 - mx), int(y1 - my), int(x2 + mx), int(y2 + my)
        if limites is not None:
            w, h = limites
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
        return x1, y1, x2, y2

    # --------------------------------------------------------------- gestos
    def _es_higa(self):
        """Puno cerrado con la punta del pulgar asomando entre indice y medio."""
        p, s = self.w3, self.escala_3d
        eje = p[9] - p[0]
        eje = eje / (np.linalg.norm(eje) + 1e-9)
        lat = p[5] - p[17]
        lat = lat - np.dot(lat, eje) * eje
        lat = lat / (np.linalg.norm(lat) + 1e-9)

        def alto(i):
            return float(np.dot(p[i] - p[0], eje) / s)

        def lado(i):
            return float(np.dot(p[i] - p[0], lat) / s)

        altura_pips = (alto(6) + alto(10)) / 2
        altura_rel = alto(4) / (altura_pips + 1e-9)
        lo, hi = sorted((lado(5), lado(9)))
        entre = lo - HIGA_MARGEN_LATERAL <= lado(4) <= hi + HIGA_MARGEN_LATERAL
        dist = float(np.linalg.norm(p[4] - (p[6] + p[10]) / 2) / s)

        self.debug.update(higa_alt=altura_rel, higa_dist=dist, higa_entre=entre)
        return altura_rel >= HIGA_ALTURA and dist <= HIGA_DISTANCIA and entre

    def clasificar(self):
        """Devuelve el nombre del gesto de esta mano (o None)."""
        e = self.estado
        cuatro_doblados = self.todos(DOB, "indice", "medio", "anular", "menique")

        # 1) Obscenos primero (tienen prioridad)
        p = self.w3
        medio_largo = np.linalg.norm(p[12] - p[0]) > 1.6 * np.linalg.norm(p[9] - p[0])
        if (e["medio"] != DOB and medio_largo
                and self.todos(DOB, "indice", "anular", "menique")):
            return "dedo_medio"
        if cuatro_doblados and not self.pulgar_ext and self._es_higa():
            return "higa"

        # 2) Like / dislike: puno cerrado + pulgar estirado vertical
        if cuatro_doblados and self.pulgar_ext:
            v = self.img[4] - self.img[2]
            v = v / (np.linalg.norm(v) + 1e-9)
            puntas_y = self.img[[6, 10, 14, 18], 1]
            if v[1] < -PULGAR_DIR and self.img[4][1] < puntas_y.min():
                return "like"
            if v[1] > PULGAR_DIR and self.img[4][1] > puntas_y.max():
                return "dislike"

        # 3) Gestos de control para dibujar
        if e["indice"] == EXT and e["medio"] != EXT and e["anular"] != EXT and e["menique"] != EXT:
            return "dibujar"
        if e["indice"] == EXT and e["medio"] == EXT and e["anular"] != EXT and e["menique"] != EXT:
            return "seleccionar"
        if self.todos(EXT, "indice", "medio", "anular", "menique"):
            return "palma"
        if cuatro_doblados:
            return "puno"
        return None


def es_corazon(m1, m2):
    """Corazon con dos manos: puntas de indices juntas arriba y pulgares juntos abajo."""
    s = (m1.escala_img + m2.escala_img) / 2
    d_ind = np.linalg.norm(m1.img[8] - m2.img[8]) / s
    d_pul = np.linalg.norm(m1.img[4] - m2.img[4]) / s
    y_ind = (m1.img[8][1] + m2.img[8][1]) / 2
    y_pul = (m1.img[4][1] + m2.img[4][1]) / 2
    alto = (y_pul - y_ind) / s
    return d_ind < CORAZON_PUNTAS and d_pul < CORAZON_PUNTAS and alto > CORAZON_ALTO
