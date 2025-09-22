# baseline_fractal_hyperchaos.py
# Simple block-based fractal compression + hyperchaotic encryption demo
# Works on small grayscale images (e.g., 128x128)

import sys, time, math
import numpy as np
from PIL import Image
import hashlib

# ---------------------------
# FRACTAL-LIKE COMPRESSION
# ---------------------------
def extract_range_blocks(img, block):
    h, w = img.shape
    coords = []
    blocks = []
    for i in range(0, h, block):
        for j in range(0, w, block):
            if i + block <= h and j + block <= w:
                coords.append((i, j))
                blocks.append(img[i:i+block, j:j+block].astype(np.float32))
    return coords, blocks

def extract_domain_blocks(img, block, scale=2, step=4):
    bh = block * scale
    h, w = img.shape
    coords = []
    blocks = []
    for i in range(0, h - bh + 1, step):
        for j in range(0, w - bh + 1, step):
            db = img[i:i+bh, j:j+bh].astype(np.float32)
            small = db.reshape(block, scale, block, scale).mean(axis=(1,3))
            coords.append((i, j))
            blocks.append(small)
    return coords, blocks

def best_affine(range_b, domain_b):
    D = domain_b.flatten()
    R = range_b.flatten()
    Dm = D.mean(); Rm = R.mean()
    denom = ((D - Dm) ** 2).sum()
    if denom == 0:
        s = 0.0
    else:
        s = ((D - Dm) * (R - Rm)).sum() / denom
    o = Rm - s * Dm
    s = max(min(s, 2.0), -2.0)
    return float(s), float(o)

def compress_fractal(img, block=8, domain_scale=2, search_step=4):
    start = time.time()
    rcoords, rblocks = extract_range_blocks(img, block)
    dcoords, dblocks = extract_domain_blocks(img, block, domain_scale, search_step)
    mappings = []
    for ri, R in enumerate(rblocks):
        best_err = float('inf')
        best_map = None
        for di, D in enumerate(dblocks):
            s, o = best_affine(R, D)
            approx = s * D + o
            err = ((approx - R) ** 2).sum()
            if err < best_err:
                best_err = err
                best_map = (rcoords[ri], dcoords[di], 0, s, o)  # rot=0 for simplicity
        mappings.append(best_map)
    print(f"[fractal compress] mapped {len(rblocks)} range-blocks in {time.time()-start:.2f}s")
    return {"shape": img.shape, "block": block, "scale": domain_scale, "mappings": mappings}

def decompress_fractal(compact, iterations=8):
    h, w = compact["shape"]
    block = compact["block"]
    mappings = compact["mappings"]
    canvas = np.zeros((h, w), dtype=np.float32)
    for it in range(iterations):
        new = canvas.copy()
        for (ri, rj), (di, dj), rot, s, o in mappings:
            bh = block * compact["scale"]
            di2 = min(di, h - bh); dj2 = min(dj, w - bh)
            domain_block = np.zeros((bh, bh), dtype=np.float32)
            domain_block[:,:] = canvas[di2:di2+bh, dj2:dj2+bh]
            small = domain_block.reshape(block, compact["scale"], block, compact["scale"]).mean(axis=(1,3))
            R = s * small + o
            new[ri:ri+block, rj:rj+block] = np.clip(R, 0, 255)
        canvas = new
    return canvas.astype(np.uint8)

# ---------------------------
# HYPERCHAOTIC ENCRYPTION (coupled logistic)
# ---------------------------
def derive_seeds(password):
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), b"seed_salt", 100000, dklen=32)
    a = int.from_bytes(key[:16], 'big') % (10**8)
    b = int.from_bytes(key[16:], 'big') % (10**8)
    x0 = 0.1234 + (a / 1e8) * 0.7
    y0 = 0.5678 + (b / 1e8) * 0.4
    return x0, y0

def hyperchaos_seq(length, x0, y0, r1=3.9999, r2=3.9876):
    x = x0; y = y0
    seq = np.empty(length, dtype=np.float64)
    for i in range(length):
        x = r1 * x * (1 - x)
        y = r2 * y * (1 - y)
        seq[i] = (x + 0.5*y) % 1.0
    return seq

def encrypt_hyperchaos(img_arr, password):
    flat = img_arr.flatten().astype(np.uint8)
    x0, y0 = derive_seeds(password)
    seq = hyperchaos_seq(len(flat), x0, y0)
    perm = np.argsort(seq)
    permuted = flat[perm]
    keystream = (np.floor(seq * 256) % 256).astype(np.uint8)
    cipher = np.bitwise_xor(permuted, keystream)
    return cipher.reshape(img_arr.shape), perm

def decrypt_hyperchaos(cipher_arr, password, perm):
    flat = cipher_arr.flatten().astype(np.uint8)
    x0, y0 = derive_seeds(password)
    seq = hyperchaos_seq(len(flat), x0, y0)
    keystream = (np.floor(seq * 256) % 256).astype(np.uint8)
    permuted = np.bitwise_xor(flat, keystream)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    recovered = permuted[inv]
    return recovered.reshape(cipher_arr.shape).astype(np.uint8)

# ---------------------------
# Utilities
# ---------------------------
def load_gray(path, resize=None):
    img = Image.open(path).convert("L")
    if resize:
        img = img.resize((resize, resize), Image.BICUBIC)
    return np.array(img)

def save_gray(arr, path):
    Image.fromarray(arr).save(path)

# ---------------------------
# Demo pipeline
# ---------------------------
def demo(inp):
    print("Load image (grayscale)...")
    img = load_gray(inp)
    print("Shape:", img.shape)

    print("Compressing (fractal-like)...")
    compact = compress_fractal(img, block=8, domain_scale=2, search_step=4)

    print("Decompressing (iterative reconstruction)...")
    recon = decompress_fractal(compact, iterations=8)
    save_gray(recon, "fractal_reconstructed.png")
    print("Saved fractal_reconstructed.png")

    # Encrypt original
    pwd = "demoPass123"
    print("Encrypting with hyperchaos (password demoPass123)...")
    cipher, perm = encrypt_hyperchaos(img, pwd)
    save_gray(cipher, "hyperchaos_encrypted.png")
    print("Saved hyperchaos_encrypted.png")

    # Decrypt
    print("Decrypting...")
    recovered = decrypt_hyperchaos(cipher, pwd, perm)
    save_gray(recovered, "hyperchaos_decrypted.png")
    print("Saved hyperchaos_decrypted.png")

    # MSE between recovered and original
    mse = np.mean((recovered.astype(np.float32) - img.astype(np.float32))**2)
    print("MSE original vs decrypted:", mse)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python baseline_fractal_hyperchaos.py input.png")
        sys.exit(1)
    demo(sys.argv[1])

#python baseline_fractal_hyperchaos.py monkey.png
# 




