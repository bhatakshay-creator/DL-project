"""
Plant Leaf Dataset Generator
============================
Generates a synthetic leaf image dataset with 10 plant species.
Each species has unique leaf shape, color, vein patterns, and textures.
Images are 64x64 RGB.

Species:
  0: Maple       - Star/palmate shape, warm colors
  1: Oak         - Lobed shape, dark green
  2: Birch       - Oval/serrated edges, light green
  3: Willow      - Long narrow ellipse, yellow-green
  4: Pine        - Needle-like thin shape, dark green
  5: Eucalyptus  - Elongated ellipse, blue-green
  6: Palm        - Fan/radiating lines, bright green
  7: Fern        - Compound/fractal shape, medium green
  8: Ivy         - Heart/triangular shape, dark green
  9: Mint        - Small oval with serrations, vibrant green
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import json

# ─────────── Configuration ───────────
IMG_SIZE = 64
SPECIES = [
    "Maple", "Oak", "Birch", "Willow", "Pine",
    "Eucalyptus", "Palm", "Fern", "Ivy", "Mint"
]
NUM_SPECIES = len(SPECIES)


# ─────────── Helper Functions ───────────
def random_bg_color(rng):
    """Generate a random light background color."""
    r = rng.randint(200, 245)
    g = rng.randint(210, 250)
    b = rng.randint(195, 240)
    return (r, g, b)


def add_vein(draw, cx, cy, angle, length, color, width=1):
    """Draw a single vein line."""
    ex = cx + length * np.cos(angle)
    ey = cy + length * np.sin(angle)
    draw.line([(cx, cy), (ex, ey)], fill=color, width=width)
    return ex, ey


def add_noise(img_array, rng, intensity=10):
    """Add slight Gaussian noise to the image."""
    noise = rng.normal(0, intensity, img_array.shape).astype(np.int16)
    noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


# ─────────── Leaf Drawing Functions ───────────
def draw_maple(draw, cx, cy, size, rng):
    """Star/palmate shape with 5-7 pointed lobes."""
    n_points = rng.choice([5, 6, 7])
    base_color = (rng.randint(180, 230), rng.randint(50, 100), rng.randint(10, 50))
    points = []
    for i in range(n_points * 2):
        angle = (i * np.pi) / n_points - np.pi / 2
        r = size if i % 2 == 0 else size * 0.4
        r += rng.uniform(-2, 2)
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=base_color, outline=(base_color[0]-40, base_color[1]-20, base_color[2]))
    # Central veins
    vein_color = (base_color[0]-60, base_color[1]-30, base_color[2])
    for i in range(n_points):
        angle = (i * 2 * np.pi) / n_points - np.pi / 2
        add_vein(draw, cx, cy, angle, size * 0.8, vein_color, width=1)


def draw_oak(draw, cx, cy, size, rng):
    """Lobed rounded shape."""
    base_color = (rng.randint(30, 70), rng.randint(100, 150), rng.randint(20, 50))
    n_lobes = rng.choice([4, 5, 6])
    points = []
    for i in range(n_lobes * 4):
        angle = (i * 2 * np.pi) / (n_lobes * 4)
        r = size * (0.7 + 0.3 * np.sin(n_lobes * angle))
        r += rng.uniform(-1, 1)
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=base_color, outline=(base_color[0]-15, base_color[1]-30, base_color[2]-10))
    # Midrib
    vein_c = (base_color[0]-20, base_color[1]-40, base_color[2]-10)
    add_vein(draw, cx, cy + size * 0.7, -np.pi/2, size * 1.3, vein_c, width=1)


def draw_birch(draw, cx, cy, size, rng):
    """Oval with serrated edges."""
    base_color = (rng.randint(100, 160), rng.randint(180, 230), rng.randint(50, 100))
    points = []
    n_pts = 40
    for i in range(n_pts):
        angle = (i * 2 * np.pi) / n_pts
        rx = size * 0.55
        ry = size * 0.85
        r = np.sqrt(1.0 / ((np.cos(angle)/rx)**2 + (np.sin(angle)/ry)**2))
        # Add serrations
        r += 2.0 * np.sin(i * 2.5)
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=base_color, outline=(base_color[0]-30, base_color[1]-40, base_color[2]-20))
    vein_c = (base_color[0]-40, base_color[1]-50, base_color[2]-20)
    add_vein(draw, cx, cy + size * 0.7, -np.pi/2, size * 1.2, vein_c, width=1)
    for s in [-1, 1]:
        for j in range(3):
            y_off = -size * 0.2 * j
            add_vein(draw, cx, cy + y_off, s * 0.6 - 0.1 * j, size * 0.35, vein_c)


def draw_willow(draw, cx, cy, size, rng):
    """Long narrow ellipse."""
    base_color = (rng.randint(120, 170), rng.randint(170, 210), rng.randint(40, 80))
    points = []
    n_pts = 36
    for i in range(n_pts):
        angle = (i * 2 * np.pi) / n_pts
        rx = size * 0.25
        ry = size * 0.95
        r = np.sqrt(1.0 / ((np.cos(angle)/rx)**2 + (np.sin(angle)/ry)**2))
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=base_color, outline=(base_color[0]-20, base_color[1]-30, base_color[2]-15))
    vein_c = (base_color[0]-30, base_color[1]-40, base_color[2]-20)
    add_vein(draw, cx, cy + size * 0.8, -np.pi/2, size * 1.5, vein_c, width=1)


def draw_pine(draw, cx, cy, size, rng):
    """Needle-like thin shape."""
    base_color = (rng.randint(20, 60), rng.randint(90, 140), rng.randint(30, 60))
    w = size * 0.12
    h = size * 0.95
    points = [
        (cx, cy - h),
        (cx + w, cy),
        (cx + w * 0.5, cy + h * 0.3),
        (cx, cy + h),
        (cx - w * 0.5, cy + h * 0.3),
        (cx - w, cy),
    ]
    # Add slight curve variation
    points = [(p[0] + rng.uniform(-1, 1), p[1] + rng.uniform(-1, 1)) for p in points]
    draw.polygon(points, fill=base_color, outline=(base_color[0]-10, base_color[1]-20, base_color[2]-10))
    vein_c = (base_color[0]-10, base_color[1]-25, base_color[2]-10)
    add_vein(draw, cx, cy + h, -np.pi/2, 2 * h * 0.85, vein_c, width=1)


def draw_eucalyptus(draw, cx, cy, size, rng):
    """Elongated ellipse, blue-green."""
    base_color = (rng.randint(60, 110), rng.randint(140, 190), rng.randint(130, 170))
    points = []
    n_pts = 36
    for i in range(n_pts):
        angle = (i * 2 * np.pi) / n_pts
        rx = size * 0.35
        ry = size * 0.85
        r = np.sqrt(1.0 / ((np.cos(angle)/rx)**2 + (np.sin(angle)/ry)**2))
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=base_color, outline=(base_color[0]-20, base_color[1]-30, base_color[2]-25))
    vein_c = (base_color[0]-25, base_color[1]-35, base_color[2]-30)
    add_vein(draw, cx, cy + size * 0.7, -np.pi/2, size * 1.3, vein_c, width=1)
    for s in [-1, 1]:
        for j in range(4):
            y_off = -size * 0.15 * j + size * 0.1
            add_vein(draw, cx, cy + y_off, s * 0.7, size * 0.25, vein_c)


def draw_palm(draw, cx, cy, size, rng):
    """Fan shape with radiating lines."""
    base_color = (rng.randint(50, 100), rng.randint(160, 210), rng.randint(40, 80))
    # Fan wedge
    n_rays = rng.choice([7, 9, 11])
    for i in range(n_rays):
        angle = -np.pi/2 + (i - n_rays//2) * 0.2
        tip_x = cx + size * 0.9 * np.cos(angle)
        tip_y = cy + size * 0.9 * np.sin(angle)
        w = size * 0.08
        left_x = cx + w * np.cos(angle + np.pi/2)
        left_y = cy + w * np.sin(angle + np.pi/2)
        right_x = cx + w * np.cos(angle - np.pi/2)
        right_y = cy + w * np.sin(angle - np.pi/2)
        shade = tuple(max(0, c + rng.randint(-15, 15)) for c in base_color)
        draw.polygon([(left_x, left_y), (tip_x, tip_y), (right_x, right_y)], fill=shade)
    # Stem
    stem_c = (base_color[0]-20, base_color[1]-40, base_color[2]-15)
    draw.line([(cx, cy), (cx, cy + size * 0.8)], fill=stem_c, width=2)


def draw_fern(draw, cx, cy, size, rng):
    """Compound shape with small leaflets."""
    base_color = (rng.randint(40, 80), rng.randint(130, 170), rng.randint(40, 70))
    stem_c = (base_color[0]-15, base_color[1]-25, base_color[2]-10)
    # Main stem
    draw.line([(cx, cy + size * 0.8), (cx, cy - size * 0.8)], fill=stem_c, width=2)
    # Leaflets
    n_pairs = rng.choice([5, 6, 7])
    for i in range(n_pairs):
        y_pos = cy - size * 0.7 + i * (size * 1.2 / n_pairs)
        leaflet_len = size * 0.3 * (1 - abs(i - n_pairs/2) / n_pairs)
        for s in [-1, 1]:
            angle = s * (0.4 + rng.uniform(-0.1, 0.1))
            ex = cx + s * leaflet_len * np.cos(angle)
            ey = y_pos + leaflet_len * np.sin(angle) * 0.3
            shade = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in base_color)
            draw.ellipse([min(cx, ex)-2, min(y_pos, ey)-1, max(cx, ex)+2, max(y_pos, ey)+1], fill=shade)


def draw_ivy(draw, cx, cy, size, rng):
    """Heart/triangular shape."""
    base_color = (rng.randint(20, 60), rng.randint(100, 150), rng.randint(30, 60))
    # Heart-like shape using points
    points = []
    n_pts = 40
    for i in range(n_pts):
        t = (i / n_pts) * 2 * np.pi
        # Heart parametric equations
        x = size * 0.5 * np.sin(t) ** 3
        y = -size * 0.45 * (
            0.8125 * np.cos(t) - 0.3125 * np.cos(2*t) -
            0.125 * np.cos(3*t) - 0.0625 * np.cos(4*t)
        )
        points.append((cx + x, cy + y))
    draw.polygon(points, fill=base_color, outline=(base_color[0]-10, base_color[1]-25, base_color[2]-10))
    vein_c = (base_color[0]-15, base_color[1]-30, base_color[2]-10)
    add_vein(draw, cx, cy + size * 0.3, -np.pi/2, size * 0.5, vein_c, width=1)
    for s in [-1, 1]:
        add_vein(draw, cx, cy, s * 0.8, size * 0.3, vein_c)
        add_vein(draw, cx, cy - size * 0.15, s * 1.1, size * 0.25, vein_c)


def draw_mint(draw, cx, cy, size, rng):
    """Small oval with pronounced serrations."""
    base_color = (rng.randint(50, 100), rng.randint(180, 230), rng.randint(60, 110))
    points = []
    n_pts = 48
    for i in range(n_pts):
        angle = (i * 2 * np.pi) / n_pts
        rx = size * 0.45
        ry = size * 0.65
        r = np.sqrt(1.0 / ((np.cos(angle)/rx)**2 + (np.sin(angle)/ry)**2))
        # Strong serrations
        r += 2.5 * np.sin(i * 4)
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=base_color, outline=(base_color[0]-20, base_color[1]-35, base_color[2]-20))
    vein_c = (base_color[0]-25, base_color[1]-40, base_color[2]-25)
    add_vein(draw, cx, cy + size * 0.5, -np.pi/2, size * 0.9, vein_c, width=1)
    for s in [-1, 1]:
        for j in range(4):
            y_off = -size * 0.12 * j + size * 0.05
            add_vein(draw, cx, cy + y_off, s * 0.55, size * 0.3, vein_c)


DRAW_FUNCTIONS = [
    draw_maple, draw_oak, draw_birch, draw_willow, draw_pine,
    draw_eucalyptus, draw_palm, draw_fern, draw_ivy, draw_mint
]


def generate_leaf_image(species_id, rng):
    """Generate a single leaf image for the given species."""
    bg_color = random_bg_color(rng)
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), bg_color)
    draw = ImageDraw.Draw(img)

    # Random position jitter and size variation
    cx = IMG_SIZE // 2 + rng.randint(-4, 5)
    cy = IMG_SIZE // 2 + rng.randint(-4, 5)
    size = rng.uniform(18, 26)

    # Random rotation
    rotation = rng.uniform(-15, 15)

    # Draw the leaf
    DRAW_FUNCTIONS[species_id](draw, cx, cy, size, rng)

    # Apply slight rotation
    img = img.rotate(rotation, resample=Image.BILINEAR, fillcolor=bg_color)

    # Apply slight blur for realism
    if rng.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Convert to numpy and add noise
    img_array = np.array(img)
    img_array = add_noise(img_array, rng, intensity=8)

    return img_array


def generate_dataset(samples_per_class=200, seed=42, output_dir=None):
    """Generate the full plant leaf dataset.

    Args:
        samples_per_class: Number of images per species
        seed: Random seed for reproducibility
        output_dir: If provided, save images to disk in folder structure

    Returns:
        X: numpy array of shape (N, 64, 64, 3), dtype uint8
        y: numpy array of shape (N,), dtype int
    """
    rng = np.random.RandomState(seed)
    total = samples_per_class * NUM_SPECIES

    X = np.zeros((total, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    y = np.zeros(total, dtype=np.int32)

    idx = 0
    for species_id in range(NUM_SPECIES):
        print(f"  Generating {samples_per_class} images for {SPECIES[species_id]}...")
        for s in range(samples_per_class):
            img = generate_leaf_image(species_id, rng)
            X[idx] = img
            y[idx] = species_id
            idx += 1

    # Shuffle
    perm = rng.permutation(total)
    X = X[perm]
    y = y[perm]

    # Optionally save to disk
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for species_id in range(NUM_SPECIES):
            species_dir = os.path.join(output_dir, SPECIES[species_id])
            os.makedirs(species_dir, exist_ok=True)

        for i in range(total):
            species_name = SPECIES[y[i]]
            img_path = os.path.join(output_dir, species_name, f"leaf_{i:04d}.png")
            Image.fromarray(X[i]).save(img_path)

        # Save metadata
        meta = {
            "species": SPECIES,
            "num_species": NUM_SPECIES,
            "samples_per_class": samples_per_class,
            "image_size": IMG_SIZE,
            "total_images": total,
            "seed": seed
        }
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  Dataset saved to {output_dir}/")

    return X, y, SPECIES


if __name__ == "__main__":
    print("Generating Plant Leaf Dataset...")
    X, y, species = generate_dataset(
        samples_per_class=200,
        output_dir="dataset/plant_leaves"
    )
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Species: {species}")
    print("Done!")
