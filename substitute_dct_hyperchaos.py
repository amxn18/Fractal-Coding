# substitute_dct_hyperchaos.py
# DCT-based compression (simple quantization) + hyperchaos encryption
# Fast and practical for demos

import sys, os
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import hashlib

def load_gray(path, resize=None):
    img = Image.open(path).convert("L")
    if resize:
        img = img.resize((resize, resize), Image.BICUBIC)
    return np.array(img)

def save_gray(arr, path):
    Image.fromarray(arr).save(path)

def block_dct_encode(img, block=8, q=10):
    h, w = img.shape
    h2 = h - (h % block)
    w2 = w - (w % block)
    img = img[:h2, :w2]
    coefs = np.zeros_like(img, dtype=np.int16)
    for i in range(0, h2, block):
        for j in range(0, w2, block):
            blk = img[i:i+block, j:j+block].astype(np.float32)
            B = dct(dct(blk.T, norm='ortho').T, norm='ortho')
            coefs[i:i+block, j:j+block] = np.round(B / q)
    return coefs, (h2, w2)

def block_dct_decode(coefs, block=8, q=10):
    h, w = coefs.shape
    out = np.zeros_like(coefs, dtype=np.float32)
    for i in range(0, h, block):
        for j in range(0, w, block):
            C = coefs[i:i+block, j:j+block].astype(np.float32) * q
            blk = idct(idct(C.T, norm='ortho').T, norm='ortho')
            out[i:i+block, j:j+block] = blk
    return np.clip(out, 0, 255).astype(np.uint8)

# reuse hyperchaos routines (seed, seq) similar to baseline
def derive_seeds(password):
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), b"dct_salt", 100000, dklen=32)
    a = int.from_bytes(key[:16], 'big') % (10**8)
    b = int.from_bytes(key[16:], 'big') % (10**8)
    x0 = 0.1111 + (a / 1e8) * 0.8
    y0 = 0.2222 + (b / 1e8) * 0.6
    return x0, y0

def hyper_seq(length, x0, y0, r1=3.9999, r2=3.9876):
    x = x0; y = y0
    seq = np.empty(length, dtype=np.float64)
    for i in range(length):
        x = r1 * x * (1 - x); y = r2 * y * (1 - y)
        seq[i] = (x + 0.5*y) % 1.0
    return seq

def encrypt_hyper_bytes(coefs_bytes, password):
    arr = np.frombuffer(coefs_bytes, dtype=np.uint8)
    x0, y0 = derive_seeds(password)
    seq = hyper_seq(len(arr), x0, y0)
    perm = np.argsort(seq)
    permuted = arr[perm]
    keystream = (np.floor(seq * 256) % 256).astype(np.uint8)
    cipher = np.bitwise_xor(permuted, keystream)
    return cipher.tobytes(), perm

def decrypt_hyper_bytes(cipher_bytes, password, perm):
    arr = np.frombuffer(cipher_bytes, dtype=np.uint8)
    x0, y0 = derive_seeds(password)
    seq = hyper_seq(len(arr), x0, y0)
    keystream = (np.floor(seq * 256) % 256).astype(np.uint8)
    permuted = np.bitwise_xor(arr, keystream)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    orig = permuted[inv]
    return orig.tobytes()

def demo(inp):
    img = load_gray(inp)
    print("Image shape:", img.shape)

    # compress (DCT quantize)
    coefs, shape = block_dct_encode(img, block=8, q=12)
    print("DCT compress done. shape:", shape)
    # bytes
    payload = coefs.tobytes()

    # encrypt bytes
    pwd = "demoPass123"
    cipher_bytes, perm = encrypt_hyper_bytes(payload, pwd)
    print("Encrypted bytes length:", len(cipher_bytes))

    # decrypt bytes
    recovered_bytes = decrypt_hyper_bytes(cipher_bytes, pwd, perm)
    rec_coefs = np.frombuffer(recovered_bytes, dtype=np.int16).reshape(shape)
    restored = block_dct_decode(rec_coefs, block=8, q=12)
    save_gray(restored, "dct_restored.png")
    print("Saved dct_restored.png")

    mse = np.mean((restored.astype(np.float32) - img[:restored.shape[0], :restored.shape[1]].astype(np.float32))**2)
    print("MSE:", mse)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python substitute_dct_hyperchaos.py input.png")
        sys.exit(1)
    demo(sys.argv[1])

# python substitute_dct_hyperchaos.py input.png

