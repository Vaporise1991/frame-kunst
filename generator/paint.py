"""Procedurele schilderijen voor Samsung The Frame (3840x2160).

Elke scène wordt eerst als eenvoudig 'onderwerp' op lage resolutie opgebouwd
en daarna met duizenden penseelstreken op een doekstructuur geschilderd.
"""
import math
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

SW, SH = 960, 540          # resolutie van de scène
OW, OH = 3840, 2160        # eindresolutie (Frame 4K)


# ---------------------------------------------------------------- helpers
def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


def stops(t, st):
    """Kleurverloop: t (array 0..1) over [(pos, '#rrggbb'), ...]."""
    t = np.clip(t, 0, 1)
    pos = np.array([p for p, _ in st])
    cols = np.array([hexc(c) for _, c in st])
    out = np.empty(t.shape + (3,))
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, cols[:, ch])
    return out


def fbm1d(n, rng, octaves=5, base=6):
    x = np.linspace(0, 1, n)
    out = np.zeros(n)
    amp = 1.0
    for o in range(octaves):
        k = base * 2 ** o
        pts = rng.normal(size=k + 2)
        out += amp * np.interp(x * k, np.arange(k + 2), pts)
        amp *= 0.5
    return out / 1.9


def fbm2d(h, w, rng, octaves=5, base=4, aspect=1.0):
    out = np.zeros((h, w))
    amp = 1.0
    for o in range(octaves):
        ky = max(2, int(base * 2 ** o))
        kx = max(2, int(base * 2 ** o * w / h / aspect))
        g = rng.normal(size=(ky + 1, kx + 1))
        out += amp * ndimage.zoom(g, (h / (ky + 1), w / (kx + 1)), order=3)[:h, :w]
        amp *= 0.5
    return out / 1.9


def grid():
    y, x = np.mgrid[0:SH, 0:SW]
    return x / SW, y / SH


def glow(img, cx, cy, r, col, strength=1.0, power=2.0):
    x, y = grid()
    d = np.sqrt(((x - cx) * SW / SH) ** 2 + (y - cy) ** 2)
    g = np.exp(-(d / r) ** power) * strength
    img += g[..., None] * (hexc(col) - img) * 0 + g[..., None] * hexc(col)
    return img


def blend(img, mask, col, a=1.0):
    m = np.clip(mask * a, 0, 1)[..., None]
    c = hexc(col) if isinstance(col, str) else col
    return img * (1 - m) + c * m


def ridge_mask(base_y, amp, rng, octaves=5, base=5):
    """Masker van alles onder een heuvelrug (y als fractie van hoogte)."""
    h = base_y + amp * fbm1d(SW, rng, octaves, base)
    _, y = grid()
    return (y >= h[None, :]).astype(float), h


def soft(mask, s=1.0):
    return ndimage.gaussian_filter(mask, s)


def to_pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def from_pil(im):
    return np.asarray(im).astype(float) / 255


def reflect(img, horizon, rng, dark=0.82, tint='#203040', ripple=3.0):
    """Spiegel alles boven de horizon in het water eronder."""
    hy = int(horizon * SH)
    out = img.copy()
    n = SH - hy
    rows = np.arange(n)
    src_rows = np.clip(hy - 1 - rows, 0, SH - 1)
    wob = fbm2d(n, SW, rng, 4, 6, 10) * ripple * (1 + rows / n * 3)[:, None]
    xs = np.clip((np.arange(SW)[None, :] + wob).astype(int), 0, SW - 1)
    refl = img[src_rows[:, None], xs]
    t = (rows / n)[:, None, None]
    refl = refl * dark * (1 - 0.25 * t) + hexc(tint) * (0.18 + 0.25 * t)
    # horizontale glinsteringen
    streak = np.clip(fbm2d(n, SW, rng, 3, 30, 14), 0, None)
    refl += (streak ** 2)[..., None] * 0.08
    out[hy:] = refl
    return out


def draw_branch(d, x, y, ang, length, width, depth, rng, col, scale):
    if depth == 0 or length < 2:
        return
    x2 = x + math.cos(ang) * length
    y2 = y - math.sin(ang) * length
    d.line([(x * scale, y * scale), (x2 * scale, y2 * scale)], fill=col,
           width=max(1, int(width * scale)))
    for _ in range(2 if depth > 2 else 3):
        draw_branch(d, x2, y2, ang + rng.uniform(-0.6, 0.6), length * rng.uniform(0.62, 0.78),
                    width * 0.65, depth - 1, rng, col, scale)


def bare_tree(d, x, y, h, rng, col, scale=1):
    draw_branch(d, x, y, math.pi / 2 + rng.uniform(-0.08, 0.08), h * 0.38, h * 0.045, 7, rng, col, scale)


def pollard(d, x, y, h, rng, col, scale=1):
    """Knotwilg: korte knoestige stam met een waaier van dunne twijgen."""
    tw = h * 0.055 + 2
    top = y - h * 0.42
    d.polygon([(x - tw, y), (x + tw, y), (x + tw * 0.8, top), (x + tw * 1.3, top - tw * 0.6),
               (x - tw * 1.3, top - tw * 0.6), (x - tw * 0.8, top)], fill=col)
    n = int(40 + h * 0.6)
    for _ in range(n):
        a = math.pi / 2 + rng.normal(0, 0.5)
        L = h * rng.uniform(0.25, 0.58)
        sx = x + rng.uniform(-tw, tw)
        sy = top - tw * 0.5
        mx = sx + math.cos(a) * L * 0.5 + rng.normal(0, h * 0.01)
        my = sy - math.sin(a) * L * 0.5
        ex, ey = sx + math.cos(a) * L, sy - math.sin(a) * L
        w = max(1, int(h / 140))
        d.line([(sx, sy), (mx, my), (ex, ey)], fill=col, width=w)


POST = []  # sterren e.d. die pas na het schilderen als stipjes worden gezet


def add_stars(rng, n, ymax):
    for _ in range(n):
        POST.append((rng.uniform(0, 1), rng.uniform(0, ymax) ** 1.2, rng.uniform(1.5, 4.5), rng.uniform(0.5, 1)))


# ---------------------------------------------------------------- scènes
def s_polder(rng):
    x, y = grid()
    H = 0.575
    img = stops(y / H, [(0, '#5d6470'), (0.45, '#9e978a'), (0.82, '#d9c4a0'), (1, '#e8dcc2')])
    clouds = fbm2d(SH, SW, rng, 5, 3, 3)
    img = blend(img, np.clip(clouds * 1.4, 0, 1) * (y < H), '#7c7e88', 0.4)
    img = glow(img, 0.66, 0.44, 0.018, '#fff1cf', 0.9, 3)
    img = glow(img, 0.66, 0.44, 0.14, '#f3cf95', 0.3)
    m, _ = ridge_mask(H - 0.012, 0.01, rng, 6, 9)
    img = blend(img, soft(m), '#646a72', 0.85)
    m2, _ = ridge_mask(H, 0.003, rng)
    field = stops(y, [(H, '#8d8a66'), (0.72, '#7b6b40'), (1, '#4c3f25')])
    patches = fbm2d(SH, SW, rng, 4, 3, 4)
    field = field * (1 + 0.12 * fbm2d(SH, SW, rng, 5, 8, 4))[..., None]
    field = field * (1 - np.clip(patches, 0, 1)[..., None] * 0.35) + hexc('#6f7a4a') * np.clip(patches, 0, 1)[..., None] * 0.35
    img = img * (1 - m2[..., None]) + field * m2[..., None]
    im = to_pil(img); d = ImageDraw.Draw(im)
    fx, fy = 0.8 * SW, H * SH + 1
    d.polygon([(fx - 22, fy), (fx + 22, fy), (fx + 22, fy - 11), (fx, fy - 26), (fx - 22, fy - 11)], fill=(78, 60, 56))
    d.rectangle([fx + 5, fy - 9, fx + 10, fy - 4], fill=(255, 205, 120))
    for k in range(5):  # verre bomenrij bij de boerderij
        tx = fx + 30 + k * 9
        d.ellipse([tx - 6, fy - 22 - k % 2 * 4, tx + 6, fy], fill=(84, 86, 92))
    img = from_pil(im)
    img = glow(img, 0.808, H - 0.012, 0.012, '#ffc070', 0.25)
    # vaart: kromt van rechts-midden op de horizon naar links-onder
    t = np.clip((y - H) / (1 - H), 0, 1)
    centre = 0.56 - 0.3 * t ** 1.3 + 0.01 * fbm1d(SH, rng)[:, None] * t
    half = 0.003 + 0.2 * t ** 1.15
    edge_n = 0.006 * fbm2d(SH, SW, rng, 4, 20, 0.3) * (0.2 + t)
    dist = np.abs(x - centre) + edge_n
    canal = (dist < half) & (y > H)
    water = reflect(img, H, rng, 0.58, '#22323a', 2.0)
    img = np.where(canal[..., None], water, img)
    bank = np.exp(-((dist - half) / (0.004 + 0.01 * t)) ** 2) * (y > H)
    img = blend(img, bank, '#3e3826', 0.7)
    mist = np.exp(-((y - H) / 0.03) ** 2) * (0.55 + 0.45 * fbm2d(SH, SW, rng, 3, 3, 5))
    img = blend(img, mist, '#e4ddd0', 0.6)
    im = to_pil(img); d = ImageDraw.Draw(im)
    refl = Image.new('RGBA', (SW, SH), (0, 0, 0, 0))
    def tree(xb, yb, h, col, seed):
        pollard(d, xb, yb, h, np.random.default_rng(seed), col)
        lay = Image.new('RGBA', (SW, SH), (0, 0, 0, 0))
        pollard(ImageDraw.Draw(lay), xb, yb, h, np.random.default_rng(seed), col + (255,))
        lay = lay.transpose(Image.FLIP_TOP_BOTTOM)
        shift = int(2 * yb - (SH - 1))
        refl.alpha_composite(lay, (0, shift) if shift >= 0 else (0, 0), (0, 0) if shift >= 0 else (0, -shift))
    def bank_x(tt, side):
        c = 0.56 - 0.3 * tt ** 1.3
        hh = 0.003 + 0.2 * tt ** 1.15
        return (c + side * (hh + 0.02 + 0.03 * tt)) * SW
    for tt in [0.06, 0.13, 0.22, 0.34, 0.5, 0.72, 0.98]:
        yy = (H + tt * (1 - H)) * SH
        h = 30 + 330 * tt ** 1.1
        tree(bank_x(tt, -1), yy, h, (46, 40, 38), int(tt * 1000))
    for tt in [0.05, 0.11, 0.2, 0.36]:
        yy = (H + tt * (1 - H)) * SH
        tree(bank_x(tt, 1), yy, 26 + 300 * tt, (58, 52, 50), int(tt * 1000) + 7)
    out = from_pil(im)
    r = np.asarray(refl).astype(float) / 255
    ra = ndimage.gaussian_filter(r[..., 3], 1.2) * canal * 0.55
    out = out * (1 - ra[..., None]) + r[..., :3] * ra[..., None]
    return out


def s_winternacht(rng):
    x, y = grid()
    img = stops(y, [(0, '#0b1230'), (0.5, '#1d2f5c'), (0.7, '#3f5b8a')])
    add_stars(rng, 260, 0.5)
    img = glow(img, 0.74, 0.2, 0.025, '#fffbe8', 1.2, 3)
    img = glow(img, 0.74, 0.2, 0.22, '#8fa6d6', 0.35)
    for i, (by, amp, col) in enumerate([(0.55, 0.06, '#9fb0d0'), (0.64, 0.05, '#c3cfe6'), (0.76, 0.05, '#dde6f4')]):
        m, h = ridge_mask(by, amp, rng, 5, 3)
        shade = stops((y - by) / 0.3, [(0, col), (1, '#7c8fb8')])
        light = 0.15 * np.gradient(-h)[None, :] * 60
        shade = shade * (1 + np.clip(light, -0.15, 0.15))[..., None]
        img = img * (1 - soft(m)[..., None]) + shade * soft(m)[..., None]
        if i < 2:
            im = to_pil(img); d = ImageDraw.Draw(im)
            for _ in range(28 if i == 0 else 16):
                px = rng.uniform(0, SW); py = h[int(px)] * SH + 3
                th = rng.uniform(14, 34) * (1.6 if i == 1 else 1)
                for k in range(5):
                    w = th * 0.32 * (1 - k / 6); yy = py - th * k / 5
                    d.polygon([(px - w, yy), (px + w, yy), (px, yy - th * 0.35)], fill=(18, 30, 48))
            img = from_pil(im)
    im = to_pil(img); d = ImageDraw.Draw(im)
    cx, cy = 0.3 * SW, 0.73 * SH
    d.rectangle([cx - 22, cy - 18, cx + 22, cy], fill=(40, 34, 44))
    d.polygon([(cx - 28, cy - 18), (cx + 28, cy - 18), (cx, cy - 40)], fill=(220, 228, 240))
    d.rectangle([cx - 8, cy - 12, cx, cy - 5], fill=(255, 196, 100))
    img = from_pil(im)
    img = glow(img, 0.296, 0.715, 0.03, '#ffb860', 0.4)
    return img


def s_rijp(rng):
    x, y = grid()
    img = stops(y / 0.62, [(0, '#9db4d4'), (0.5, '#e7c3c8'), (0.85, '#f6c99a'), (1, '#fde3b8')])
    img = glow(img, 0.36, 0.6, 0.035, '#fff4d6', 1.3, 3)
    img = glow(img, 0.36, 0.6, 0.3, '#ffcf9c', 0.35)
    m, h = ridge_mask(0.62, 0.008, rng, 4, 4)
    frost = stops((y - 0.62) / 0.38, [(0, '#d8cfe0'), (0.4, '#ece8f2'), (1, '#c9d2e6')])
    frost *= (1 + 0.06 * fbm2d(SH, SW, rng, 6, 10, 4))[..., None]
    img = img * (1 - m[..., None]) + frost * m[..., None]
    im = to_pil(img); d = ImageDraw.Draw(im)
    trees = sorted(rng.uniform(0.02, 0.98, 9))
    for tx in trees:
        px = tx * SW; py = h[int(px)] * SH + 2
        th = rng.uniform(80, 150)
        # lange blauwe schaduw naar de kijker toe
        d.polygon([(px - 2, py), (px + 2, py), (px + (px - 0.36 * SW) * 0.9, SH), (px + (px - 0.36 * SW) * 0.9 - 8, SH)],
                  fill=(150, 160, 200))
        bare_tree(d, px, py, th, rng, (70, 62, 80))
    return from_pil(im)


def s_dorp(rng):
    x, y = grid()
    img = stops(y, [(0, '#0f1d3d'), (0.45, '#2c4a7a'), (0.62, '#c98a6a')])
    add_stars(rng, 110, 0.3)
    im = to_pil(img); d = ImageDraw.Draw(im)
    base = 0.66 * SH
    xx = -10
    windows = []
    while xx < SW + 10:
        w = rng.uniform(34, 60); hh = rng.uniform(60, 110)
        col = tuple(int(c) for c in rng.uniform([22, 22, 36], [36, 34, 50]))
        if abs(xx - 0.6 * SW) < 30:
            # kerktoren met spits
            d.rectangle([xx, base - 210, xx + 40, base], fill=(26, 26, 40))
            d.polygon([(xx - 4, base - 210), (xx + 44, base - 210), (xx + 20, base - 320)], fill=(26, 26, 40))
            windows.append((xx + 14, base - 170, xx + 26, base - 150))
            xx += 40
            continue
        d.rectangle([xx, base - hh, xx + w, base], fill=col)
        # trapgevel
        steps = 4
        for k in range(steps):
            inset = w * 0.5 * k / steps
            d.rectangle([xx + inset, base - hh - 12 * (k + 1), xx + w - inset, base - hh - 12 * k], fill=col)
        # sneeuw op de gevel
        d.line([(xx, base - hh), (xx + w, base - hh)], fill=(210, 220, 240), width=2)
        for r in range(int(hh // 30)):
            for c in range(max(1, int(w // 22))):
                if rng.random() < 0.55:
                    wx = xx + 8 + c * 22; wy = base - hh + 12 + r * 30
                    windows.append((wx, wy, wx + 9, wy + 13))
        xx += w + rng.uniform(0, 4)
    for wbox in windows:
        d.rectangle(wbox, fill=tuple(int(c) for c in rng.uniform([235, 160, 70], [255, 210, 120])))
    d.rectangle([0, base, SW, SH], fill=(196, 206, 228))
    img = from_pil(im)
    # warm licht op de sneeuw en een zachte gloed per raam
    snow = (y > 0.66)
    img = blend(img, snow * fbm2d(SH, SW, rng, 4, 5, 3).clip(0) , '#9aa9cc', 0.6)
    for (a, b, c, e) in windows[:: max(1, len(windows) // 60)]:
        img = glow(img, (a + c) / 2 / SW, (b + e) / 2 / SH, 0.035, '#ffb35c', 0.18)
    img = blend(img, np.exp(-((y - 0.7) / 0.04) ** 2), '#e8a86a', 0.12)
    return img


def s_bloesem(rng):
    x, y = grid()
    img = stops(y, [(0, '#7fa8d9'), (0.45, '#c6dcf0')])
    cl = fbm2d(SH, SW, rng, 5, 2, 3)
    img = blend(img, np.clip((cl - 0.1) * 2.2, 0, 1) * (y < 0.5), '#fbfbf7', 0.9)
    for by, amp, col in [(0.48, 0.05, '#8fb08a'), (0.56, 0.04, '#7fae5c'), (0.66, 0.03, '#98c25a')]:
        m, _ = ridge_mask(by, amp, rng, 5, 3)
        tex = hexc(col) * (1 + 0.1 * fbm2d(SH, SW, rng, 5, 6, 3))[..., None]
        img = img * (1 - soft(m)[..., None]) + tex * soft(m)[..., None]
    im = to_pil(img); d = ImageDraw.Draw(im)
    # boomgaard in rijen, achter naar voor
    for row, (ry, sc) in enumerate([(0.6, 0.5), (0.7, 0.75), (0.84, 1.05), (1.02, 1.45)]):
        n = int(9 / sc) + 2
        for k in range(n):
            tx = (k + 0.5 + rng.uniform(-0.2, 0.2) + (row % 2) * 0.5) / n * SW
            ty = ry * SH
            th = 90 * sc
            d.line([(tx, ty), (tx, ty - th * 0.5)], fill=(80, 60, 55), width=max(2, int(5 * sc)))
            for _ in range(int(170 * sc)):
                a = rng.uniform(0, 2 * math.pi); r = abs(rng.normal(0, 0.5)) * th * 0.5
                px = tx + math.cos(a) * r * 1.2; py = ty - th * 0.62 + math.sin(a) * r * 0.75
                rr = rng.uniform(2, 5) * sc
                c = [(250, 220, 230), (244, 190, 208), (255, 244, 248), (230, 160, 190)][rng.integers(4)]
                d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=c)
    # bloemetjes in het gras
    for _ in range(900):
        px = rng.uniform(0, SW); py = rng.uniform(0.7, 1) * SH
        rr = (py / SH - 0.6) * 4
        d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(250, 240, 180) if rng.random() < 0.6 else (255, 255, 255))
    return from_pil(im)


def s_koren(rng):
    x, y = grid()
    img = stops(y, [(0, '#3d7fc4'), (0.5, '#a8cde8')])
    cl = fbm2d(SH, SW, rng, 5, 2, 2.5)
    puffs = np.clip((cl - 0.25) * 2.5, 0, 1) * np.clip(1 - y / 0.45, 0, 1)
    img = blend(img, puffs, '#ffffff', 0.95)
    img = blend(img, puffs * np.clip(fbm2d(SH, SW, rng, 3, 3), 0, 1), '#b9c2d4', 0.5)
    img = glow(img, 0.82, 0.12, 0.035, '#fff3b0', 0.9, 3)
    img = glow(img, 0.82, 0.12, 0.25, '#ffe9a8', 0.3)
    m, _ = ridge_mask(0.5, 0.015, rng, 6, 10)
    img = blend(img, soft(m), '#4f6f73', 1)
    m2, _ = ridge_mask(0.53, 0.01, rng, 4, 3)
    wheat = stops((y - 0.53) / 0.47, [(0, '#d9b75a'), (0.5, '#e3ad3e'), (1, '#b97d22')])
    streaks = fbm2d(SH, SW, rng, 4, 4, 0.12)  # verticale structuur
    wheat *= (1 + 0.14 * streaks + 0.08 * fbm2d(SH, SW, rng, 4, 3, 3))[..., None]
    img = img * (1 - m2[..., None]) + wheat * m2[..., None]
    im = to_pil(img); d = ImageDraw.Draw(im)
    for _ in range(1600):
        py = rng.uniform(0.55, 1) ** 0.7 * SH
        if py < 0.55 * SH:
            continue
        px = rng.uniform(0, SW)
        k = (py / SH - 0.5) * 2
        if rng.random() < 0.12:
            rr = 1 + 6 * k
            d.ellipse([px - rr, py - rr, px + rr, py + rr * 0.8], fill=(205, 40, 30))
        else:
            d.line([(px, py), (px + rng.normal(0, 2), py - 26 * k)], fill=(240, 205, 120), width=max(1, int(2 * k)))
    return from_pil(im)


def s_kust(rng):
    x, y = grid()
    img = stops(y / 0.45, [(0, '#5f7488'), (0.6, '#a9b7c2'), (1, '#dfe2df')])
    cl = fbm2d(SH, SW, rng, 5, 2, 3)
    img = blend(img, np.clip(cl * 1.5, 0, 1) * (y < 0.45), '#eef0f0', 0.7)
    img = blend(img, np.clip(-cl * 1.5, 0, 1) * (y < 0.45), '#4e6070', 0.5)
    img = glow(img, 0.25, 0.3, 0.12, '#fff5df', 0.35)
    sea = (y >= 0.45) & (y < 0.62)
    seacol = stops((y - 0.45) / 0.17, [(0, '#4f6a7a'), (1, '#7d968f')])
    seacol *= (1 + 0.1 * fbm2d(SH, SW, rng, 4, 30, 10))[..., None]
    img = np.where(sea[..., None], seacol, img)
    img = blend(img, np.exp(-((y - 0.6) / 0.006) ** 2) * (fbm2d(SH, SW, rng, 3, 8, 6) > -0.2), '#f2f3ee', 0.8)
    beach = y >= 0.62
    sand = stops((y - 0.62) / 0.38, [(0, '#b8ab90'), (1, '#d9c9a2')])
    sand *= (1 + 0.05 * fbm2d(SH, SW, rng, 5, 5, 3))[..., None]
    img = np.where(beach[..., None], sand, img)
    # duin rechts
    dune = (y > 0.4 + 0.55 * np.clip((1 - x) / 0.55, 0, 1) ** 2.2 + 0.015 * fbm1d(SW, rng, 5, 6)[None, :]) & (x > 0.42)
    dune_col = stops(x, [(0.45, '#cbb88f'), (1, '#a79772')])
    img = np.where(dune[..., None], dune_col * (1 + 0.06 * fbm2d(SH, SW, rng, 4, 5))[..., None], img)
    im = to_pil(img); d = ImageDraw.Draw(im)
    top = np.argmax(dune, axis=0)
    for _ in range(2600):
        px = rng.uniform(0.45, 1) * SW
        ty = top[int(px)]
        if ty == 0:
            continue
        py = ty + abs(rng.normal(0, 40))
        L = rng.uniform(12, 30) * (py / SH)
        a = math.pi / 2 + rng.normal(-0.25, 0.25)
        c = tuple(int(v) for v in rng.uniform([120, 125, 80], [175, 170, 110]))
        d.line([(px, py), (px + math.cos(a) * L, py - math.sin(a) * L)], fill=c, width=1)
    # meeuwen
    for _ in range(4):
        gx, gy, s = rng.uniform(0.2, 0.6) * SW, rng.uniform(0.15, 0.3) * SH, rng.uniform(4, 7)
        d.line([(gx - s, gy - s * 0.4), (gx, gy), (gx + s, gy - s * 0.4)], fill=(60, 66, 74), width=2)
    return from_pil(im)


def s_kleurveld(rng):
    x, y = grid()
    img = np.ones((SH, SW, 3)) * hexc('#6b2a24')
    n = fbm2d(SH, SW, rng, 5, 6, 1)
    def rect(img, x0, x1, y0, y1, col, a=0.92):
        e = 0.012
        mx = np.clip((x - x0) / e, 0, 1) * np.clip((x1 - x) / e, 0, 1)
        my = np.clip((y - y0) / (e * 1.6), 0, 1) * np.clip((y1 - y) / (e * 1.6), 0, 1)
        mask = mx * my * (0.85 + 0.15 * n)
        return blend(img, ndimage.gaussian_filter(mask, 3), col, a)
    img = rect(img, 0.07, 0.93, 0.07, 0.42, '#d8892f')
    img = rect(img, 0.07, 0.93, 0.47, 0.6, '#3a1f3a')
    img = rect(img, 0.07, 0.93, 0.65, 0.93, '#a5401f')
    img *= (1 + 0.08 * fbm2d(SH, SW, rng, 6, 8, 2))[..., None]
    return img


def s_noorderlicht(rng):
    x, y = grid()
    img = stops(y, [(0, '#050a1c'), (0.55, '#10264a')])
    add_stars(rng, 320, 0.45)
    # gordijnen: een golvende curve met verticale strepen erboven
    curve = 0.32 + 0.1 * fbm1d(SW, rng, 3, 2) + 0.05 * np.sin(np.linspace(0, 5, SW))
    dist = curve[None, :] - y
    streaks = np.clip(fbm2d(SH, SW, rng, 3, 2, 0.05), 0, None)
    band = np.where(dist > 0, np.exp(-dist / 0.16), np.exp(dist / 0.012)) * (0.5 + streaks)
    band *= (y < 0.55)
    green = band * np.clip(1 - dist * 3, 0, 1)
    purple = band * np.clip(dist * 3, 0, 1)
    img += green[..., None] * hexc('#41f0a0') * 0.9 + purple[..., None] * hexc('#9a5cf0') * 0.55
    m, h = ridge_mask(0.5, 0.08, rng, 5, 3)
    img = blend(img, soft(m), '#070d18', 1)
    img = reflect(img, 0.56, rng, 0.7, '#061020', 2.5)
    m2, _ = ridge_mask(0.9, 0.02, rng, 4, 3)
    img = blend(img, m2, '#03070e', 1)
    return img


def s_herfstbos(rng):
    x, y = grid()
    img = stops(y, [(0, '#e9c78a'), (0.5, '#c98a4a'), (1, '#5a3a22')])
    img = glow(img, 0.5, 0.4, 0.18, '#ffe2a6', 0.3)
    im = to_pil(img); d = ImageDraw.Draw(im)
    pal = [(214, 96, 36), (232, 150, 40), (196, 60, 30), (240, 190, 70), (150, 70, 30), (120, 110, 40)]
    for layer, (n, wmin, wmax, dark) in enumerate([(26, 4, 8, 0.75), (16, 8, 16, 0.55), (9, 16, 30, 0.35)]):
        for _ in range(n):
            tx = rng.uniform(0, SW)
            if abs(tx - SW / 2) < 60 + layer * 40:
                continue
            w = rng.uniform(wmin, wmax)
            tc = tuple(int(c * (1 - dark) + h * dark) for c, h in zip((48, 32, 22), (214, 160, 110)))
            d.rectangle([tx - w / 2, 0, tx + w / 2, SH], fill=tc)
        for _ in range(1800 + layer * 600):
            px = rng.uniform(0, SW); py = rng.uniform(-0.05, 0.6) * SH
            if ((px - SW / 2) / (45 + 25 * np.sin(py / 23))) ** 2 + ((py - 0.45 * SH) / 110) ** 2 < 1:
                continue
            r = rng.uniform(3, 7) * (1 + layer * 0.6)
            c = pal[rng.integers(len(pal))]
            c = tuple(int(v * (1 - dark) + h * dark) for v, h in zip(c, (230, 180, 120)))
            d.ellipse([px - r, py - r * 0.8, px + r, py + r * 0.8], fill=c)
    # bospad
    d.polygon([(SW * 0.47, SH * 0.6), (SW * 0.53, SH * 0.6), (SW * 0.75, SH), (SW * 0.25, SH)], fill=(196, 140, 80))
    for _ in range(1400):
        px = rng.uniform(0, SW); py = rng.uniform(0.62, 1) * SH
        r = 2 + (py / SH - 0.6) * 9
        c = pal[rng.integers(len(pal))]
        d.ellipse([px - r, py - r * 0.6, px + r, py + r * 0.6], fill=c)
    img = from_pil(im)
    # zonnestralen
    beams = np.zeros((SH, SW))
    ang = np.arctan2(y - 0.0, x - 0.55)
    beams = (np.sin(ang * 40 + 2) > 0.6) * np.clip(1 - y, 0, 1) * 0.25
    img = blend(img, soft(beams, 4), '#fff3cf', 0.5)
    return img


SCENES = [
    ('01_late_herfst_polder', s_polder),
    ('02_winternacht_maanlicht', s_winternacht),
    ('03_rijp_bij_zonsopkomst', s_rijp),
    ('04_december_avond_dorp', s_dorp),
    ('05_lente_bloesem', s_bloesem),
    ('06_zomer_korenveld', s_koren),
    ('07_noordzee_duinen', s_kust),
    ('08_kleurveld_herfsttinten', s_kleurveld),
    ('09_noorderlicht_meer', s_noorderlicht),
    ('10_herfstbos', s_herfstbos),
]


# ---------------------------------------------------------------- schilderen
def paint(scene, rng):
    base = to_pil(scene).resize((OW, OH), Image.BICUBIC)
    src = np.asarray(base).astype(np.int16)
    canvas = base.filter(ImageFilter.GaussianBlur(14))
    d = ImageDraw.Draw(canvas)

    lum = scene @ np.array([0.3, 0.59, 0.11])
    lum = ndimage.gaussian_filter(lum, 1.5)
    gy = ndimage.sobel(lum, 0); gx = ndimage.sobel(lum, 1)
    mag = np.hypot(gx, gy)
    flow = fbm2d(SH, SW, rng, 3, 2) * 0.35
    angle = np.where(mag > 0.02, np.arctan2(gy, gx) + math.pi / 2, flow)
    # hoekveld gladstrijken via cos/sin
    ca = ndimage.gaussian_filter(np.cos(2 * angle), 3); sa = ndimage.gaussian_filter(np.sin(2 * angle), 3)
    angle = 0.5 * np.arctan2(sa, ca)
    magn = mag / (mag.max() + 1e-9)

    sx = OW / SW
    passes = [(34, 95, 7000, 0.0), (18, 52, 22000, 0.0), (9, 26, 45000, 0.25), (4, 12, 45000, 0.6)]
    for width, length, count, edge_bias in passes:
        px = rng.uniform(0, OW, count); py = rng.uniform(0, OH, count)
        ix = np.clip((px / sx).astype(int), 0, SW - 1); iy = np.clip((py / sx).astype(int), 0, SH - 1)
        if edge_bias:
            keep = rng.random(count) < (1 - edge_bias) + edge_bias * np.clip(magn[iy, ix] * 8, 0, 1)
            px, py, ix, iy = px[keep], py[keep], ix[keep], iy[keep]
        ang = angle[iy, ix] + rng.normal(0, 0.18, len(px))
        L = length * rng.uniform(0.6, 1.3, len(px))
        cols = src[np.clip(py.astype(int), 0, OH - 1), np.clip(px.astype(int), 0, OW - 1)].astype(float)
        cols *= rng.normal(1, 0.035, (len(px), 1))
        cols += rng.normal(0, 5, (len(px), 3))
        cols = np.clip(cols, 0, 255).astype(int)
        dx = np.cos(ang) * L / 2; dy = np.sin(ang) * L / 2
        for i in range(len(px)):
            w = max(1, int(width * rng.uniform(0.7, 1.2)))
            c = tuple(cols[i])
            d.line([(px[i] - dx[i], py[i] - dy[i]), (px[i] + dx[i], py[i] + dy[i])], fill=c, width=w)
            if w > 6:
                r = w / 2
                d.ellipse([px[i] + dx[i] - r, py[i] + dy[i] - r, px[i] + dx[i] + r, py[i] + dy[i] + r], fill=c)

    for (fx, fy, r, b) in POST:
        cx, cy = fx * OW, fy * OH
        base_c = np.asarray(canvas.getpixel((int(min(cx, OW - 1)), int(min(cy, OH - 1)))), float)
        for k, (rr, a) in enumerate([(r * 3, 0.15), (r * 1.6, 0.35), (r, 1.0)]):
            c = tuple(int(v) for v in base_c * (1 - a * b) + np.array([255, 250, 235]) * a * b)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=c)
    out = np.asarray(canvas).astype(float) / 255
    # doekstructuur + lichte vignettering
    yy, xx = np.mgrid[0:OH, 0:OW]
    weave = 1 - 0.025 * (np.sin(xx * 1.9) * np.sin(yy * 1.9) + 1) / 2
    grain = 1 + rng.normal(0, 0.012, (OH, OW))
    vig = 1 - 0.18 * (((xx / OW - 0.5) * 1.6) ** 2 + ((yy / OH - 0.5) * 1.6) ** 2)
    out *= (weave * grain * vig)[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


if __name__ == '__main__':
    which = sys.argv[1:] or [n for n, _ in SCENES]
    for i, (name, fn) in enumerate(SCENES):
        if name not in which and name[:2] not in which:
            continue
        rng = np.random.default_rng(1000 + i)
        POST.clear()
        scene = np.clip(fn(rng), 0, 1)
        to_pil(scene).save(f'/home/claude/frame/out/{name}_scene.png')
        img = paint(scene, np.random.default_rng(2000 + i))
        img.save(f'/home/claude/frame/out/{name}.jpg', quality=92)
        img.resize((960, 540), Image.LANCZOS).save(f'/home/claude/frame/out/{name}_preview.jpg', quality=88)
        print('klaar', name, flush=True)
