#!/usr/bin/env python3
"""
svg2scad_simplified.py

Versione modificata di svg2scad.py con semplificazione poligonale (Ramer-Douglas-Peucker)
Aggiunte:
  - rdp() + remove_collinear() + simplify_glyphs()
  - opzione CLI --tolerance per controllare l'aggressività della semplificazione
  - mantiene le altre opzioni originali: --samples-per-unit, --circle-samples, --scale, --out-dir, -o

Dipendenze (installare se necessario):
    pip install svgpathtools lxml

Esempio d'uso:
    python svg2scad_simplified.py input.svg --samples-per-unit 0.8 --circle-samples 24 --tolerance 0.5 --out-dir ./scad

"""
import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from svgpathtools import parse_path
except Exception as e:
    raise ImportError("This script requires svgpathtools. Install with: pip install svgpathtools")

NUM_DEFAULT_SAMPLES = 16

# ----------------------- helper utilities -----------------------

def sanitize_name(name: str) -> str:
    if not name:
        return None
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    if re.match(r"^[0-9]", s):
        s = "g_" + s
    return s or None

def parse_floats(s: str):
    nums = re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", s)
    return [float(x) for x in nums]

def apply_transform(point, transform_attr: str):
    """Applica trasformazioni semplici: translate(tx,ty) e scale(sx,sy)."""
    x, y = point
    if not transform_attr:
        return x, y
    m = re.search(r"translate\s*\(\s*([\-\d.eE+,\s]+)\)", transform_attr)
    if m:
        vals = parse_floats(m.group(1))
        if len(vals) == 1:
            tx = vals[0]; ty = 0.0
        else:
            tx, ty = vals[0], vals[1]
        x += tx; y += ty
    m = re.search(r"scale\s*\(\s*([\-\d.eE+,\s]+)\)", transform_attr)
    if m:
        vals = parse_floats(m.group(1))
        if len(vals) == 1:
            sx = vals[0]; sy = vals[0]
        else:
            sx, sy = vals[0], vals[1]
        x *= sx; y *= sy
    return x, y

# ----------------------- polyline simplification (Ramer–Douglas–Peucker) -----------------------

def perp_dist(pt, a, b):
    """Distanza perpendicolare del punto pt rispetto al segmento ab."""
    (x, y) = pt
    (x1, y1) = a
    (x2, y2) = b
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x - x1, y - y1)
    t = ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)
    projx = x1 + t * dx
    projy = y1 + t * dy
    return math.hypot(x - projx, y - projy)

def rdp(points, epsilon):
    """Semplificazione Ramer–Douglas–Peucker."""
    if not points or len(points) < 3:
        return list(points)
    dmax = 0.0
    index = -1
    a = points[0]
    b = points[-1]
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], a, b)
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        left = rdp(points[:index+1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [a, b]

def remove_collinear(points, tol=1e-12):
    """Rimuove punti collineari consecutivi."""
    if len(points) < 3:
        return points[:]
    out = [points[0]]
    for p in points[1:]:
        while len(out) >= 2:
            a = out[-2]; b = out[-1]; c = p
            area2 = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
            if area2 <= tol:
                out.pop()
            else:
                break
        out.append(p)
    return out

def simplify_glyphs(glyphs, epsilon=0.5, min_points=3):
    """Applica RDP a tutti i sottopercorsi delle glyphs."""
    new_glyphs = []
    for g in glyphs:
        new_subpaths = []
        for sp in g['subpaths']:
            if not sp:
                continue
            closed = (len(sp) > 1 and sp[0] == sp[-1])
            pts = list(sp)
            if closed:
                pts = pts[:-1]
            simp = rdp(pts, epsilon)
            simp = remove_collinear(simp, tol=1e-12)
            if closed:
                if len(simp) >= 1 and simp[0] != simp[-1]:
                    simp.append(simp[0])
            if len(simp) < min_points:
                continue
            new_subpaths.append(simp)
        new_glyphs.append({'name': g['name'], 'subpaths': new_subpaths})
    return new_glyphs

# ----------------------- SVG element samplers -----------------------

def sample_path_d(d_attr: str, samples_per_unit: float = 1.0):
    p = parse_path(d_attr)
    try:
        length = p.length()
    except Exception:
        length = 0.0
        for seg in p:
            try:
                length += seg.length()
            except Exception:
                pass
    n = max(int(math.ceil(length * samples_per_unit)), len(p) * 6, NUM_DEFAULT_SAMPLES)
    pts = []
    for i in range(n + 1):
        t = i / n
        c = p.point(t)
        pts.append((c.real, c.imag))
    return pts

def parse_points_attr(s: str):
    vals = parse_floats(s)
    pts = []
    for i in range(0, len(vals), 2):
        if i + 1 < len(vals):
            pts.append((vals[i], vals[i + 1]))
    return pts

def sample_circle(cx, cy, r, samples=64):
    pts = []
    for i in range(samples):
        a = 2 * math.pi * i / samples
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts

def sample_ellipse(cx, cy, rx, ry, samples=64):
    pts = []
    for i in range(samples):
        a = 2 * math.pi * i / samples
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return pts

# ----------------------- main parser -----------------------

def parse_svg(svg_path: Path, samples_per_unit: float = 1.0, circle_samples: int = 48):
    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    ns = {k if k else 'svg': v for k, v in root.attrib.items() if k.startswith('xmlns')}
    
    vb = root.get('viewBox')
    if vb:
        vb_vals = parse_floats(vb)
        if len(vb_vals) >= 4:
            vb_minx, vb_miny, vb_w, vb_h = vb_vals[0], vb_vals[1], vb_vals[2], vb_vals[3]
        else:
            vb_minx = vb_miny = vb_w = vb_h = None
    else:
        vb_minx = vb_miny = vb_w = vb_h = None
        h_attr = root.get('height')
        w_attr = root.get('width')
        if h_attr and w_attr:
            try:
                vb_w = float(re.findall(r"[\d.]+", w_attr)[0])
                vb_h = float(re.findall(r"[\d.]+", h_attr)[0])
                vb_minx = 0.0; vb_miny = 0.0
            except Exception:
                vb_w = vb_h = None
                
    glyphs = []
    idx = 0
    
    # Nuova logica: gestisci glifi composti
    compound_glyphs = {}
    for elem in root.findall('.//{http://www.w3.org/2000/svg}symbol'):
        gid = elem.get('id')
        if not gid:
            continue
        compound_glyphs[gid] = list(elem.iter())

    all_elements = list(root.iter())
    for _, elements in compound_glyphs.items():
        all_elements.extend(elements)

    for elem in all_elements:
        tag = elem.tag
        if '}' in tag:
            tag = tag.split('}', 1)[1]
        
        if tag not in ('path', 'polygon', 'polyline', 'rect', 'circle', 'ellipse'):
            continue
        
        gid = elem.get('id') or elem.get('inkscape:label') or f'g{idx}'
        name = sanitize_name(gid) or f'g{idx}'
        
        if not name:
            name = f'glyph_{idx}'

        transform_attr = elem.get('transform')

        subpaths = []

        if tag == 'path':
            d = elem.get('d')
            if not d: continue
            pts = sample_path_d(d, samples_per_unit=samples_per_unit)
            pts = [apply_transform(p, transform_attr) for p in pts]
            subpaths.append(pts)

        elif tag in ('polygon', 'polyline'):
            pts = parse_points_attr(elem.get('points', ''))
            pts = [apply_transform(p, transform_attr) for p in pts]
            subpaths.append(pts)

        elif tag == 'rect':
            x = float(elem.get('x', '0'))
            y = float(elem.get('y', '0'))
            w = float(elem.get('width', '0'))
            h = float(elem.get('height', '0'))
            pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            pts = [apply_transform(p, transform_attr) for p in pts]
            subpaths.append(pts)

        elif tag == 'circle':
            cx = float(elem.get('cx', '0'))
            cy = float(elem.get('cy', '0'))
            r = float(elem.get('r', '0'))
            pts = sample_circle(cx, cy, r, samples=circle_samples)
            pts = [apply_transform(p, transform_attr) for p in pts]
            subpaths.append(pts)

        elif tag == 'ellipse':
            cx = float(elem.get('cx', '0'))
            cy = float(elem.get('cy', '0'))
            rx = float(elem.get('rx', '0'))
            ry = float(elem.get('ry', '0'))
            pts = sample_ellipse(cx, cy, rx, ry, samples=circle_samples)
            pts = [apply_transform(p, transform_attr) for p in pts]
            subpaths.append(pts)

        if vb_h is not None:
            new_sub = []
            for sp in subpaths:
                newsp = []
                for (x, y) in sp:
                    ny = vb_miny + vb_h - (y - vb_miny)
                    newsp.append((x, ny))
                new_sub.append(newsp)
            subpaths = new_sub
        
        if subpaths:
            if any(g['name'] == name for g in glyphs):
                continue
            glyphs.append({'name': name, 'subpaths': subpaths})

        idx += 1

    return glyphs

# ----------------------- OpenSCAD writer -----------------------

def scad_point_str(p, fmt='{:.3f}'):
    return '[%s, %s]' % (fmt.format(p[0]), fmt.format(p[1]))

def generate_scad(glyphs, out_path: Path, scale: float = 1.0, polygon_epsilon: float = 0.001):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('// Generated by svg2scad_simplified.py\n')
        f.write('// Modules: one module per SVG element (named glyph_<id>)\n\n')

        for g in glyphs:
            name = g['name']
            modname = f'glyph_{name}'
            f.write(f'module {modname}(scale_factor = {scale}) {{\n')
            f.write('    union() {\n')
            for sp in g['subpaths']:
                if not sp: continue
                pts = [[p[0] * scale, p[1] * scale] for p in sp]
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                f.write('        polygon(points = [\n')
                for p in pts:
                    f.write('            %s,\n' % scad_point_str(p))
                f.write('        ]);\n')
            f.write('    }\n')
            f.write('}\n\n')

        f.write('// End of file\n')


# ----------------------- CLI -----------------------

def main():
    parser = argparse.ArgumentParser(description='Convert simple SVG shapes into OpenSCAD glyph_... modules')
    parser.add_argument('input', help='Input SVG file')
    parser.add_argument('-o', '--out', default=None, help='Output SCAD file (default: input.scad)')
    parser.add_argument('-d', '--out-dir', default=None, help='Directory where to save the output .scad (will be created if missing)')
    parser.add_argument('--samples-per-unit', type=float, default=1, help='Sampling density for paths (samples per SVG unit of length)')
    parser.add_argument('--circle-samples', type=int, default=48, help='Number of samples for circles/ellipses')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale output coordinates by this factor')
    parser.add_argument('--tolerance', type=float, default=0.2, help='Simplification tolerance (epsilon) in SVG units; higher => fewer points')

    args = parser.parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        parser.error(f"Input file not found: {in_path}")

    if args.out:
        out_path = Path(args.out)
    elif args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / in_path.with_suffix('.scad').name
    else:
        out_path = in_path.with_suffix('.scad')

    glyphs = parse_svg(in_path, samples_per_unit=args.samples_per_unit, circle_samples=args.circle_samples)

    if args.tolerance is not None and args.tolerance > 0:
        glyphs = simplify_glyphs(glyphs, epsilon=args.tolerance)

    generate_scad(glyphs, out_path, scale=args.scale)
    
    print(f"Wrote {len(glyphs)} glyphs to {out_path}")


if __name__ == '__main__':
    main()