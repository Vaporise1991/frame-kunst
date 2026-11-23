"""Tien herfstschilderijen, gebouwd op de helpers en penseel-render uit paint.py."""
import math
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from paint import (SW, SH, hexc, stops, fbm1d, fbm2d, grid, glow, blend, ridge_mask, soft,
                   to_pil, from_pil, reflect, bare_tree, paint, add_stars, POST)

AUT = [(214, 96, 36), (232, 150, 40), (196, 60, 30), (240, 190, 70), (150, 70, 30), (170, 120, 40), (120, 40, 30)]


def leafy_tree(d, x, y, h, rng, pal=AUT, dens=1.0, trunk=(52, 38, 30), haze=None, hz=0.0, spread=0.55):
    def mix(c):
        return tuple(int(v * (1 - hz) + w * hz) for v, w in zip(c, haze)) if haze else c
    tw = max(1, h * 0.035)
    d.polygon([(x - tw, y), (x + tw, y), (x + tw * 0.4, y - h * 0.6), (x - tw * 0.4, y - h * 0.6)], fill=mix(trunk))
    for _ in range(5):
        a = math.pi / 2 + rng.uniform(-0.9, 0.9)
        sy = y - h * rng.uniform(0.35, 0.6)
        d.line([(x, sy), (x + math.cos(a) * h * 0.3, sy - math.sin(a) * h * 0.3)], fill=mix(trunk), width=max(1, int(tw * 0.6)))
    cx, cy = x, y - h * 0.68
    for _ in range(int(260 * dens * max(0.4, h / 120))):
        r = abs(rng.normal(0, 0.45)) * h * spread
        a = rng.uniform(0, 2 * math.pi)
        px, py = cx + math.cos(a) * r, cy + math.sin(a) * r * 0.75
        s = rng.uniform(0.02, 0.05) * h
        c = mix(pal[rng.integers(len(pal))])
        d.ellipse([px - s, py - s * 0.8, px + s, py + s * 0.8], fill=c)


# ------------------------------------------------------------------ scènes
def h_beukenbos(rng):
    x, y = grid()
    img = stops(y, [(0, '#d8c6a4'), (0.6, '#b88a55'), (1, '#4e3420')])
    im = to_pil(img); d = ImageDraw.Draw(im)
    for layer, (n, hz) in enumerate([(30, 0.7), (18, 0.45), (10, 0.15)]):
        for _ in range(n):
            tx = rng.uniform(-20, SW + 20)
            w = (3 + layer * 6) * rng.uniform(0.7, 1.3)
            col = tuple(int(c * (1 - hz) + h * hz) for c, h in zip((70, 66, 62), (220, 200, 170)))
            d.rectangle([tx - w / 2, 0, tx + w / 2, SH], fill=col)
        for _ in range(1500 + 500 * layer):
            px = rng.uniform(0, SW); py = rng.uniform(-0.05, 0.5) * SH
            r = rng.uniform(3, 7) * (1 + 0.5 * layer)
            c = AUT[rng.integers(len(AUT))]
            c = tuple(int(v * (1 - hz) + h * hz) for v, h in zip(c, (225, 200, 160)))
            d.ellipse([px - r, py - r * 0.8, px + r, py + r * 0.8], fill=c)
    for _ in range(2500):  # bladertapijt
        px = rng.uniform(0, SW); py = rng.uniform(0.7, 1) * SH
        r = 1.5 + (py / SH - 0.65) * 10
        c = AUT[rng.integers(len(AUT))]
        d.ellipse([px - r, py - r * 0.5, px + r, py + r * 0.5], fill=c)
    img = from_pil(im)
    ang = np.arctan2(y + 0.2, x - 0.3)
    beams = (np.sin(ang * 34) > 0.55) * np.clip(1 - y * 1.1, 0, 1)
    img = blend(img, soft(beams.astype(float), 5), '#fff1cc', 0.35)
    mist = np.exp(-((y - 0.62) / 0.12) ** 2) * (0.6 + 0.4 * fbm2d(SH, SW, rng, 3, 3, 4))
    return blend(img, mist, '#efe2c8', 0.45)


def h_molen(rng):
    x, y = grid()
    H = 0.66
    img = stops(y / H, [(0, '#3b2a4a'), (0.35, '#a34a3a'), (0.7, '#ec8a3c'), (1, '#f7c56a')])
    cl = fbm2d(SH, SW, rng, 5, 3, 4)
    img = blend(img, np.clip(cl * 1.6, 0, 1) * (y < H * 0.8), '#5a2d3c', 0.55)
    img = glow(img, 0.3, H - 0.02, 0.03, '#fff0b0', 1.0, 3)
    img = glow(img, 0.3, H - 0.02, 0.3, '#ffb060', 0.35)
    m, _ = ridge_mask(H, 0.004, rng)
    img = blend(img, m, '#2a1a1e', 1)
    img = reflect(img, H + 0.01, rng, 0.75, '#2a1a24', 2.5)
    im = to_pil(img); d = ImageDraw.Draw(im)
    d.rectangle([0, int((H + 0.01) * SH) - 3, SW, int((H + 0.01) * SH) + 2], fill=(34, 22, 26))
    mx, my = 0.68 * SW, H * SH
    col = (30, 20, 24)
    d.polygon([(mx - 30, my), (mx + 30, my), (mx + 18, my - 120), (mx - 18, my - 120)], fill=col)
    d.polygon([(mx - 22, my - 120), (mx + 22, my - 120), (mx, my - 150)], fill=col)
    d.rectangle([mx - 50, my - 40, mx + 50, my - 34], fill=col)  # stelling
    hub = (mx, my - 128)
    for k in range(4):
        a = math.radians(20 + k * 90)
        ex, ey = hub[0] + math.cos(a) * 140, hub[1] - math.sin(a) * 140
        d.line([hub, (ex, ey)], fill=col, width=5)
        px, py = -math.sin(a) * 14, -math.cos(a) * 14
        for t in np.linspace(0.25, 1, 7):
            bx, by = hub[0] + math.cos(a) * 140 * t, hub[1] - math.sin(a) * 140 * t
            d.line([(bx, by), (bx + px, by + py)], fill=col, width=2)
        d.line([(hub[0] + math.cos(a) * 35 + px, hub[1] - math.sin(a) * 35 + py), (ex + px, ey + py)], fill=col, width=2)
    for _ in range(6):  # vogels
        gx, gy, s = rng.uniform(0.35, 0.6) * SW, rng.uniform(0.2, 0.4) * SH, rng.uniform(3, 6)
        d.line([(gx - s, gy - s * 0.4), (gx, gy), (gx + s, gy - s * 0.4)], fill=(40, 24, 30), width=2)
    for _ in range(900):  # riet op de voorgrond
        px = rng.uniform(0, SW); py = SH - abs(rng.normal(0, 25))
        L = rng.uniform(20, 70)
        a = math.pi / 2 + rng.normal(0, 0.15)
        d.line([(px, py), (px + math.cos(a) * L, py - math.sin(a) * L)], fill=(26, 16, 20), width=2)
    return from_pil(im)


def h_vijver(rng):
    x, y = grid()
    H = 0.52
    img = stops(y / H, [(0, '#6f97c4'), (1, '#d9e4ea')])
    cl = fbm2d(SH, SW, rng, 5, 2, 3)
    img = blend(img, np.clip((cl - 0.05) * 2, 0, 1) * (y < H), '#ffffff', 0.85)
    im = to_pil(img); d = ImageDraw.Draw(im)
    for layer, (n, base, hz) in enumerate([(30, 0.5, 0.45), (22, 0.53, 0.2), (12, 0.56, 0.0)]):
        for _ in range(n):
            tx = rng.uniform(-30, SW + 30)
            h = rng.uniform(70, 140) * (1 + layer * 0.3)
            leafy_tree(d, tx, base * SH + 4, h, rng, dens=1.0, haze=(200, 210, 220), hz=hz)
    img = from_pil(im)
    img = reflect(img, H + 0.06, rng, 0.78, '#1e3040', 2.5)
    im = to_pil(img); d = ImageDraw.Draw(im)
    for _ in range(140):  # drijvende bladeren
        px = rng.uniform(0, SW); py = rng.uniform(0.66, 1) * SH
        s = 1 + (py / SH - 0.6) * 9
        d.ellipse([px - s, py - s * 0.45, px + s, py + s * 0.45], fill=AUT[rng.integers(len(AUT))])
    return from_pil(im)


def h_gracht(rng):
    x, y = grid()
    H = 0.6
    img = stops(y / H, [(0, '#7d8794'), (1, '#b8b4ae')])
    im = to_pil(img); d = ImageDraw.Draw(im)
    xx = -10
    while xx < SW + 10:
        w = rng.uniform(46, 72); hh = rng.uniform(150, 230)
        base = H * SH - 12
        col = tuple(int(c) for c in rng.uniform([70, 44, 36], [130, 80, 60]))
        d.rectangle([xx, base - hh, xx + w, base], fill=col)
        top = base - hh
        kind = rng.integers(3)
        if kind == 0:
            d.polygon([(xx, top), (xx + w, top), (xx + w / 2, top - w * 0.55)], fill=col)
        elif kind == 1:
            for k in range(4):
                ins = w * 0.5 * k / 4
                d.rectangle([xx + ins, top - 14 * (k + 1), xx + w - ins, top - 14 * k], fill=col)
        else:
            d.rectangle([xx + w * 0.25, top - 24, xx + w * 0.75, top], fill=col)
            d.ellipse([xx + w * 0.25, top - 34, xx + w * 0.75, top - 14], fill=col)
        for r in range(int(hh // 34)):
            for c in range(max(1, int(w // 20))):
                wx = xx + 6 + c * 20; wy = top + 14 + r * 34
                lit = rng.random() < 0.3
                d.rectangle([wx, wy, wx + 10, wy + 18], fill=(250, 200, 120) if lit else (60, 64, 72))
        xx += w + 2
    d.rectangle([0, H * SH - 12, SW, H * SH], fill=(70, 62, 58))  # kade
    for k in range(6):  # iepen langs de kade
        tx = (k + 0.5) * SW / 6 + rng.uniform(-30, 30)
        leafy_tree(d, tx, H * SH - 10, rng.uniform(95, 130), rng, dens=0.55, spread=0.45,
                   pal=[(222, 160, 40), (236, 190, 70), (200, 120, 30), (180, 140, 50)])
    img = from_pil(im)
    img = reflect(img, H, rng, 0.7, '#283038', 3.5)
    # regen
    im = to_pil(img); d = ImageDraw.Draw(im)
    for _ in range(1400):
        px = rng.uniform(0, SW); py = rng.uniform(0, SH)
        L = rng.uniform(8, 20)
        d.line([(px, py), (px - L * 0.2, py + L)], fill=(205, 210, 215), width=1)
    img = from_pil(im)
    return blend(img, np.ones((SH, SW)), '#8d96a2', 0.12)


def h_heuvels(rng):
    x, y = grid()
    img = stops(y, [(0, '#f1d9a6'), (0.35, '#f4c98a')])
    img = glow(img, 0.2, 0.18, 0.3, '#fff2cc', 0.4)
    im = to_pil(img); d = ImageDraw.Draw(im)
    layers = [(0.36, 0.05, (176, 150, 150), 0.65), (0.46, 0.06, (150, 120, 90), 0.45),
              (0.58, 0.07, (140, 120, 60), 0.25), (0.72, 0.07, (120, 110, 50), 0.05)]
    img = from_pil(im)
    for i, (by, amp, col, hz) in enumerate(layers):
        m, h = ridge_mask(by, amp, rng, 5, 2 + i)
        base = np.array(col) / 255 * (1 - hz) + hexc('#f0d8b0') * hz
        tex = base * (1 + 0.1 * fbm2d(SH, SW, rng, 4, 6, 3))[..., None]
        img = img * (1 - soft(m)[..., None]) + tex * soft(m)[..., None]
        im = to_pil(img); d = ImageDraw.Draw(im)
        for _ in range(int(14 + i * 10)):
            tx = rng.uniform(0, SW)
            ty = h[int(min(tx, SW - 1))] * SH + rng.uniform(4, 40 + i * 30)
            leafy_tree(d, tx, ty, rng.uniform(20, 40) * (1 + i * 0.8), rng, dens=0.6,
                       haze=(240, 216, 176), hz=hz)
        img = from_pil(im)
    im = to_pil(img); d = ImageDraw.Draw(im)
    # kronkelweg op de voorste heuvel
    pts = [(0.62 * SW + 80 * math.sin(t * 5) * t, 0.7 * SH + t * 0.3 * SH) for t in np.linspace(0, 1, 40)]
    for k, (px, py) in enumerate(pts):
        w = 2 + k * 0.6
        d.ellipse([px - w, py - w * 0.4, px + w, py + w * 0.4], fill=(214, 196, 160))
    return from_pil(im)


def h_paddenstoelen(rng):
    x, y = grid()
    img = stops(y, [(0, '#3d4a2a'), (0.5, '#6d5a32'), (1, '#3a2716')])
    img = glow(img, 0.6, 0.2, 0.35, '#f0d090', 0.45)
    im = to_pil(img); d = ImageDraw.Draw(im)
    for _ in range(3500):  # mos en bladeren
        px = rng.uniform(0, SW); py = rng.uniform(0.45, 1) * SH
        r = rng.uniform(2, 6) * (0.5 + py / SH)
        c = AUT[rng.integers(len(AUT))] if rng.random() < 0.6 else (88, 110, 50)
        d.ellipse([px - r, py - r * 0.6, px + r, py + r * 0.6], fill=c)
    # omgevallen stam
    d.polygon([(0, 0.62 * SH), (SW * 0.55, 0.56 * SH), (SW * 0.55, 0.66 * SH), (0, 0.74 * SH)], fill=(84, 62, 44))
    d.ellipse([SW * 0.53, 0.56 * SH, SW * 0.58, 0.66 * SH], fill=(168, 128, 88))
    for _ in range(300):
        px = rng.uniform(0, SW * 0.54); py = 0.6 * SH - px / SW * 0.1 * SH + rng.uniform(-6, 30)
        r = rng.uniform(2, 4)
        d.ellipse([px - r, py - r, px + r, py + r], fill=(96, 130, 56))
    def mushroom(mx, my, s, cap=(196, 40, 30), dots=True):
        d.rounded_rectangle([mx - s * 0.18, my - s * 0.9, mx + s * 0.18, my], s * 0.1, fill=(236, 226, 206))
        d.pieslice([mx - s * 0.6, my - s * 1.25, mx + s * 0.6, my - s * 0.45], 180, 360, fill=cap)
        d.rectangle([mx - s * 0.6, my - s * 0.86, mx + s * 0.6, my - s * 0.82], fill=(220, 200, 170))
        if dots:
            for _ in range(7):
                dx = rng.uniform(-0.45, 0.45) * s; dy = rng.uniform(-1.15, -0.92) * s
                if (dx / (s * 0.6)) ** 2 + ((dy + 0.85 * s) / (s * 0.4)) ** 2 < 0.8:
                    r = s * 0.05
                    d.ellipse([mx + dx - r, my + dy - r, mx + dx + r, my + dy + r], fill=(250, 245, 235))
    for mx, my, s in [(0.62, 0.86, 120), (0.74, 0.9, 80), (0.7, 0.8, 55), (0.2, 0.92, 70), (0.3, 0.88, 45)]:
        mushroom(mx * SW, my * SH, s)
    for mx, my, s in [(0.45, 0.74, 34), (0.48, 0.76, 26), (0.88, 0.82, 40)]:
        mushroom(mx * SW, my * SH, s, cap=(150, 96, 50), dots=False)
    return from_pil(im)


def h_laan(rng):
    x, y = grid()
    img = stops(y, [(0, '#c9d6de'), (0.5, '#e8dcc4')])
    im = to_pil(img); d = ImageDraw.Draw(im)
    vx, vy = 0.5 * SW, 0.5 * SH
    d.polygon([(vx - 3, vy), (vx + 3, vy), (SW * 0.85, SH), (SW * 0.15, SH)], fill=(150, 120, 92))
    for i in range(16, 0, -1):  # bomen van ver naar dichtbij
        t = (i / 16) ** 2.2
        s = 1 - t
        h = 14 + 330 * s ** 1.5
        yb = vy + (SH - vy) * s ** 1.3
        for side in (-1, 1):
            xb = vx + side * (10 + (SW * 0.5) * s ** 1.2)
            hz = 0.6 * t
            leafy_tree(d, xb, yb, h, rng, dens=0.8, spread=0.42, haze=(230, 214, 186), hz=hz)
    for _ in range(3000):
        px = rng.uniform(0, SW); py = rng.uniform(0.52, 1) * SH
        r = 1 + (py / SH - 0.5) * 9
        d.ellipse([px - r, py - r * 0.5, px + r, py + r * 0.5], fill=AUT[rng.integers(len(AUT))])
    img = from_pil(im)
    img = glow(img, 0.5, 0.48, 0.12, '#fff3d8', 0.5)
    for _ in range(3):  # fietser in de verte
        pass
    return img


def h_storm(rng):
    x, y = grid()
    H = 0.55
    img = stops(y / H, [(0, '#2b3038'), (0.6, '#5a6470'), (1, '#9aa3a8')])
    cl = fbm2d(SH, SW, rng, 6, 2, 2)
    img = blend(img, np.clip(cl * 1.5, 0, 1) * (y < H), '#1d2128', 0.7)
    img = blend(img, np.clip(-cl * 1.5, 0, 1) * (y < H), '#c9cfd2', 0.4)
    ang = np.arctan2(y - 0.05, x - 0.7)
    rays = (np.sin(ang * 20) > 0.7) * (y < H) * (y > 0.15)
    img = blend(img, soft(rays.astype(float), 6), '#e8ecec', 0.25)
    sea = y >= H
    t = (y - H) / (1 - H)
    seacol = stops(t, [(0, '#3c4c52'), (1, '#1e2a2e')])
    waves = fbm2d(SH, SW, rng, 5, 16, 7)
    seacol = seacol * (1 + 0.25 * waves)[..., None]
    img = np.where(sea[..., None], seacol, img)
    foam = ((waves > 0.55) & (waves < 0.62)) * sea + (waves > 0.75) * sea
    img = blend(img, soft(foam.astype(float), 1.2), '#e6ecea', 0.8)
    im = to_pil(img); d = ImageDraw.Draw(im)
    # rots met vuurtoren
    d.polygon([(SW * 0.62, H * SH + 10), (SW * 0.7, H * SH - 30), (SW * 0.86, H * SH - 40), (SW * 0.95, H * SH + 12)], fill=(30, 30, 34))
    lx, ly = SW * 0.79, H * SH - 40
    d.polygon([(lx - 14, ly), (lx + 14, ly), (lx + 8, ly - 120), (lx - 8, ly - 120)], fill=(230, 228, 222))
    for k in range(3):
        yy = ly - 20 - k * 36
        d.polygon([(lx - 13 + k * 2, yy), (lx + 13 - k * 2, yy), (lx + 12 - k * 2, yy - 14), (lx - 12 + k * 2, yy - 14)], fill=(170, 40, 36))
    d.rectangle([lx - 10, ly - 136, lx + 10, ly - 120], fill=(255, 232, 160))
    d.polygon([(lx - 12, ly - 136), (lx + 12, ly - 136), (lx, ly - 150)], fill=(40, 40, 44))
    img = from_pil(im)
    img = glow(img, lx / SW, (ly - 128) / SH, 0.04, '#fff0b0', 0.8)
    # lichtbundel
    beam = (np.abs((y - (ly - 128) / SH) - (x - lx / SW) * -0.18) < 0.004 + (lx / SW - x).clip(0) * 0.06) * (x < lx / SW)
    img = blend(img, soft(beam.astype(float), 3), '#fff6d0', 0.35)
    # opspattend water
    spray = np.exp(-((x - 0.66) / 0.04) ** 2 - ((y - 0.5) / 0.05) ** 2) * (fbm2d(SH, SW, rng, 4, 8) > 0)
    return blend(img, spray, '#f2f4f2', 0.6)


def h_oogstmaan(rng):
    x, y = grid()
    H = 0.62
    img = stops(y / H, [(0, '#0e1430'), (0.6, '#2c2a50'), (1, '#6a4a5a')])
    add_stars(rng, 150, 0.4)
    img = glow(img, 0.34, 0.4, 0.075, '#f9b25a', 1.6, 6)
    img = glow(img, 0.34, 0.4, 0.3, '#c9784a', 0.35)
    m, h = ridge_mask(H, 0.012, rng, 5, 4)
    field = stops((y - H) / (1 - H), [(0, '#3a2e30'), (0.5, '#5a4632'), (1, '#2e2218')])
    rows = (np.sin((x - 0.5) / (y - H + 0.02).clip(0.02) * 6) > 0.3) * m
    field = field * (1 + 0.18 * rows)[..., None]
    img = img * (1 - m[..., None]) + field * m[..., None]
    moonlight = np.exp(-((x - 0.34) / (0.02 + (y - H).clip(0) * 0.6)) ** 2) * m * 0.35
    img = blend(img, moonlight, '#d89a5a', 1)
    im = to_pil(img); d = ImageDraw.Draw(im)
    for k, (hx, hy, s) in enumerate([(0.62, 0.7, 34), (0.76, 0.76, 48), (0.5, 0.82, 64), (0.86, 0.9, 90), (0.18, 0.9, 80)]):
        px, py = hx * SW, hy * SH
        r = s * 0.55
        d.rectangle([px - r * 0.2, py - 2 * r, px + r * 1.3, py], fill=(92, 66, 40))
        d.ellipse([px - r, py - 2 * r, px + r, py], fill=(150, 108, 58))
        for q in (0.75, 0.5, 0.25):
            d.ellipse([px - r * q, py - r - r * q, px + r * q, py - r + r * q], outline=(108, 76, 42), width=2)
        d.ellipse([px - r * 0.9, py - 1.9 * r, px - r * 0.1, py - 1.1 * r], fill=(196, 140, 72))
    for k in range(6):
        tx = rng.uniform(0.55, 0.98) * SW
        bare_tree(d, tx, h[int(tx)] * SH + 2, rng.uniform(40, 80), rng, (20, 18, 30))
    return from_pil(im)


def h_bladeren(rng):
    x, y = grid()
    img = stops(x * 0.6 + y * 0.4, [(0, '#2e1a2c'), (0.5, '#5a2a26'), (1, '#8a4a22')])
    img *= (1 + 0.1 * fbm2d(SH, SW, rng, 5, 3))[..., None]
    im = to_pil(img); d = ImageDraw.Draw(im)
    pal = [(232, 120, 40), (246, 182, 60), (196, 50, 36), (250, 214, 120), (164, 72, 30), (220, 90, 60)]
    for _ in range(46):
        cx, cy = rng.uniform(0, SW), rng.uniform(0, SH)
        s = rng.uniform(25, 75) * rng.choice([0.6, 1, 1.5])
        a = rng.uniform(0, 2 * math.pi)
        c = pal[rng.integers(len(pal))]
        lobes = rng.integers(3, 6)
        pts = []
        for k in range(60):  # esdoornachtig blad: lobben via een sinus op de straal
            t = k / 60 * 2 * math.pi
            r = s * (0.55 + 0.45 * abs(math.sin(t * lobes / 2))) * (0.9 if math.sin(t) < -0.8 else 1)
            pts.append((cx + math.cos(t + a) * r, cy + math.sin(t + a) * r))
        d.polygon(pts, fill=c)
        dark = tuple(int(v * 0.6) for v in c)
        for k in range(lobes):
            t = (k + 0.5) / lobes * 2 * math.pi
            d.line([(cx, cy), (cx + math.cos(t + a) * s * 0.85, cy + math.sin(t + a) * s * 0.85)], fill=dark, width=2)
        d.line([(cx, cy), (cx - math.cos(a + 0.3) * s * 1.2, cy - math.sin(a + 0.3) * s * 1.2)], fill=dark, width=3)
    return from_pil(im)


SCENES = [
    ('11_herfst_beukenbos_mist', h_beukenbos),
    ('12_herfst_molen_zonsondergang', h_molen),
    ('13_herfst_vijver', h_vijver),
    ('14_herfst_gracht_regen', h_gracht),
    ('15_herfst_heuvels', h_heuvels),
    ('16_herfst_paddenstoelen', h_paddenstoelen),
    ('17_herfst_bomenlaan', h_laan),
    ('18_herfst_storm_vuurtoren', h_storm),
    ('19_herfst_oogstmaan', h_oogstmaan),
    ('20_herfst_bladeren_abstract', h_bladeren),
]

if __name__ == '__main__':
    which = sys.argv[1:]
    for i, (name, fn) in enumerate(SCENES):
        if which and name[:2] not in which:
            continue
        rng = np.random.default_rng(3000 + i)
        POST.clear()
        scene = np.clip(fn(rng), 0, 1)
        img = paint(scene, np.random.default_rng(4000 + i))
        img.save(f'/home/claude/frame/out2/{name}.jpg', quality=92)
        img.resize((960, 540), Image.LANCZOS).save(f'/home/claude/frame/out2/{name}_preview.jpg', quality=88)
        print('klaar', name, flush=True)
