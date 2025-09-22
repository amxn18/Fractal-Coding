"""
fractal_with_hyperchaos.py

End-to-end demo:
  - encode:  image -> fractal compression (mappings) -> serialize -> hyperchaos encrypt -> save .enc
  - decode: .enc -> hyperchaos decrypt -> deserialize -> fractal decompress -> save reconstructed image

Notes:
 - Use small grayscale images (e.g., 128x128) for speed.
 - This is a demo/proof-of-concept (not optimized).
 - Dependencies: pillow, numpy, scipy
    pip install pillow numpy scipy
"""

import sys
import os
import time
import pickle
import math
import hashlib
import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------------------
# Basic image helpers
# ---------------------------
def load_grayscale(path, resize=None):
    img = Image.open(path).convert("L")
    if resize:
        img = img.resize((resize, resize), Image.BICUBIC)
    return np.array(img, dtype=np.float32)

def save_grayscale(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)

# ---------------------------
# Fractal compression functions (adapted from your code)
# ---------------------------

def reduce_block(img, factor):
    """Downsample by averaging non-overlapping factor x factor blocks."""
    h_out = img.shape[0] // factor
    w_out = img.shape[1] // factor
    out = np.zeros((h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            out[i, j] = np.mean(img[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
    return out

def rotate_block(img, angle):
    return ndimage.rotate(img, angle, reshape=False, mode='nearest')

def flip_block(img, direction):
    # direction = 1 -> no flip, -1 -> vertical flip
    if direction == 1:
        return img
    else:
        return img[::-1, :]

def apply_transformation(block, direction, angle, contrast=1.0, brightness=0.0):
    t = flip_block(block, direction)
    t = rotate_block(t, angle)
    return contrast * t + brightness

def find_contrast_brightness_ls(D, S):
    """
    Solve for s,o in s*S + o ~= D in least squares (two-parameter linear fit).
    Returns (s, o).
    """
    S_flat = S.reshape(-1)
    D_flat = D.reshape(-1)
    A = np.vstack([S_flat, np.ones_like(S_flat)]).T  # columns: S, ones
    # Solve A * [s, o] = D
    x, _, _, _ = np.linalg.lstsq(A, D_flat, rcond=None)
    s = float(x[0])
    o = float(x[1])
    # clamp s to avoid extreme values (optional)
    s = max(min(s, 4.0), -4.0)
    return s, o

def generate_domain_blocks(img, source_size, dest_size, step):
    """Extract domain blocks of size source_size and downsample them to dest_size."""
    factor = source_size // dest_size
    h, w = img.shape
    domain_blocks = []
    domain_coords = []
    for i in range(0, h - source_size + 1, step):
        for j in range(0, w - source_size + 1, step):
            src = img[i:i+source_size, j:j+source_size].astype(np.float32)
            # downsample by averaging factor x factor patches
            down = src.reshape(dest_size, factor, dest_size, factor).mean(axis=(1,3))
            domain_blocks.append(down)
            domain_coords.append((i, j))
    return domain_coords, domain_blocks

def extract_range_blocks(img, dest_size):
    coords = []
    blocks = []
    h, w = img.shape
    for i in range(0, h, dest_size):
        for j in range(0, w, dest_size):
            if i + dest_size <= h and j + dest_size <= w:
                coords.append((i, j))
                blocks.append(img[i:i+dest_size, j:j+dest_size].astype(np.float32))
    return coords, blocks

def compress_fractal(img, source_size=16, dest_size=8, step=8, transforms=None):
    """
    For each non-overlapping destination (range) block, find best transformed domain block.
    Store mapping: (range_coord, domain_coord, direction, angle, s, o)
    """
    if transforms is None:
        directions = [1, -1]
        angles = [0, 90, 180, 270]
        transforms = [(d, a) for d in directions for a in angles]

    r_coords, r_blocks = extract_range_blocks(img, dest_size)
    d_coords, d_blocks = generate_domain_blocks(img, source_size, dest_size, step)

    mappings = []
    t0 = time.time()
    # For each range block, find best domain block + transform + affine params
    for ridx, R in enumerate(r_blocks):
        best_err = float('inf')
        best_map = None
        for didx, D in enumerate(d_blocks):
            for direction, angle in transforms:
                S = apply_transformation(D, direction, angle, contrast=1.0, brightness=0.0)
                s, o = find_contrast_brightness_ls(R, S)
                approx = s * S + o
                err = np.sum((R - approx)**2)
                if err < best_err:
                    best_err = err
                    best_map = (r_coords[ridx], d_coords[didx], direction, angle, s, o)
        mappings.append(best_map)
    t1 = time.time()
    print(f"[compress] mapped {len(r_blocks)} range-blocks in {t1-t0:.2f}s")
    compact = dict(shape=img.shape, source_size=source_size, dest_size=dest_size, step=step, mappings=mappings)
    return compact

def decompress_fractal(compact, iterations=8):
    shape = compact["shape"]
    source_size = compact["source_size"]
    dest_size = compact["dest_size"]
    step = compact["step"]
    mappings = compact["mappings"]

    h, w = shape
    canvas = np.random.randint(0, 256, (h, w)).astype(np.float32)  # start from noise
    for it in range(iterations):
        new_canvas = canvas.copy()
        for mapping in mappings:
            (ri, rj), (di, dj), direction, angle, s, o = mapping
            # build domain block from current canvas
            bh = source_size
            # ensure indices in range
            di2 = min(di, max(0, h - bh))
            dj2 = min(dj, max(0, w - bh))
            domain_block = canvas[di2:di2+bh, dj2:dj2+bh]
            # downsample
            factor = source_size // dest_size
            small = domain_block.reshape(dest_size, factor, dest_size, factor).mean(axis=(1,3))
            S = apply_transformation(small, direction, angle, contrast=1.0, brightness=0.0)
            R_recon = s * S + o
            new_canvas[ri:ri+dest_size, rj:rj+dest_size] = np.clip(R_recon, 0, 255)
        canvas = new_canvas
    return canvas.astype(np.uint8)

# ---------------------------
# Hyperchaotic encryption (serialize -> permute -> xor)
# ---------------------------

def derive_two_seeds(password, salt=b"fractal_hyper_salt"):
    """Return two floats in (0,1) derived from password using PBKDF2."""
    key = hashlib.pbkdf2_hmac("sha256", password.encode('utf-8'), salt, 100000, dklen=32)
    a = int.from_bytes(key[:16], 'big') % (10**8)
    b = int.from_bytes(key[16:], 'big') % (10**8)
    x0 = 0.123456 + (a / 1e8) * 0.7
    y0 = 0.654321 + (b / 1e8) * 0.7
    return float(x0), float(y0)

def hyperchaos_sequence(length, x0, y0, r1=3.9999, r2=3.9876):
    """Generate length floats in [0,1) using two coupled logistic maps."""
    x = x0; y = y0
    seq = np.empty(length, dtype=np.float64)
    for i in range(length):
        x = r1 * x * (1.0 - x)
        y = r2 * y * (1.0 - y)
        seq[i] = (x + 0.5*y) % 1.0
    return seq

def encrypt_bytes_hyperchaos(plain_bytes, password):
    """Encrypt bytes: produce permutation from seq, permute, then XOR with keystream."""
    data = np.frombuffer(plain_bytes, dtype=np.uint8)
    x0, y0 = derive_two_seeds(password)
    seq = hyperchaos_sequence(len(data), x0, y0)
    perm = np.argsort(seq)         # ascending order -> permutation
    permuted = data[perm]
    keystream = (np.floor(seq * 256) % 256).astype(np.uint8)
    cipher = np.bitwise_xor(permuted, keystream)
    return cipher.tobytes(), perm.astype(np.int32)  # return perm to allow inversion

def decrypt_bytes_hyperchaos(cipher_bytes, password, perm):
    data = np.frombuffer(cipher_bytes, dtype=np.uint8)
    x0, y0 = derive_two_seeds(password)
    seq = hyperchaos_sequence(len(data), x0, y0)
    keystream = (np.floor(seq * 256) % 256).astype(np.uint8)
    permuted = np.bitwise_xor(data, keystream)
    # invert permutation
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    original = permuted[inv]
    return original.tobytes()

# ---------------------------
# Serialization helpers
# ---------------------------

def save_compact_to_file(compact, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_compact_from_bytes(b):
    return pickle.loads(b)

# ---------------------------
# CLI operations
# ---------------------------

def do_encode(input_image_path, password, out_enc_path):
    print("Loading image:", input_image_path)
    img = load_grayscale(input_image_path)
    print("Image shape:", img.shape)

    print("Running fractal compression (this can be slow)...")
    compact = compress_fractal(img, source_size=16, dest_size=8, step=8)

    print("Serializing compressed mappings...")
    ser = pickle.dumps(compact, protocol=pickle.HIGHEST_PROTOCOL)

    print("Encrypting serialized data with hyperchaos...")
    cipher_bytes, perm = encrypt_bytes_hyperchaos(ser, password)

    # Save encrypted data and permutation (we need perm to decrypt; in practice you'd store perm securely)
    with open(out_enc_path, 'wb') as f:
        # file format: 4 bytes perm length n, n*4 bytes of perm (int32), then ciphertext
        # but storing perm is for demo convenience; in a real hyperchaos system you would
        # regenerate perm from the password alone because it is deterministic from seed.
        # Here we still write perm so that decode can work even if small length rounding differences occur.
        n = len(perm)
        f.write(n.to_bytes(4, 'big'))
        f.write(perm.tobytes())
        f.write(cipher_bytes)
    print("Wrote encrypted file:", out_enc_path)
    print("Done.")

def do_decode(enc_path, password, out_image_path):
    print("Reading encrypted file:", enc_path)
    with open(enc_path, 'rb') as f:
        n_bytes = f.read(4)
        if len(n_bytes) < 4:
            raise ValueError("Invalid file format")
        n = int.from_bytes(n_bytes, 'big')
        perm_bytes = f.read(n * 4)
        perm = np.frombuffer(perm_bytes, dtype=np.int32)
        cipher = f.read()
    print("Read perm length:", n, "cipher bytes:", len(cipher))

    print("Decrypting...")
    plain_bytes = decrypt_bytes_hyperchaos(cipher, password, perm)

    print("Deserializing compressed mapping...")
    compact = load_compact_from_bytes(plain_bytes)
    print("Compact shape:", compact.get('shape'))

    print("Decompressing (iterative reconstruction)...")
    recon = decompress_fractal(compact, iterations=10)
    save_grayscale(recon, out_image_path)
    print("Saved reconstructed image:", out_image_path)

# ---------------------------
# Entry point
# ---------------------------
def usage():
    print("Usage:")
    print("  Encode: python fractal_with_hyperchaos.py encode <input_image> <password> <out.enc>")
    print("  Decode: python fractal_with_hyperchaos.py decode <in.enc> <password> <out_image.png>")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "encode":
        if len(sys.argv) != 5:
            usage(); sys.exit(1)
        _, _, inp, pwd, out = sys.argv
        do_encode(inp, pwd, out)
    elif cmd == "decode":
        if len(sys.argv) != 5:
            usage(); sys.exit(1)
        _, _, inp_enc, pwd, out_img = sys.argv
        do_decode(inp_enc, pwd, out_img)
    else:
        usage()
        sys.exit(1)
