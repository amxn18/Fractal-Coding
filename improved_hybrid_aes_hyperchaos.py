# improved_hybrid_aes_hyperchaos.py
# DCT-based compression + AES-GCM encryption + hyperchaos permutation
# Supports color images (RGB)

import sys, os
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import hashlib

# ---------- DCT helpers ----------
def img_to_channels(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return arr  # shape H,W,3

def block_dct_encode_channel(chan, block=8, q=12):
    h, w = chan.shape
    h2 = h - (h % block); w2 = w - (w % block)
    chan = chan[:h2, :w2]
    coefs = np.zeros_like(chan, dtype=np.int16)
    for i in range(0, h2, block):
        for j in range(0, w2, block):
            b = chan[i:i+block, j:j+block].astype(np.float32)
            B = dct(dct(b.T, norm='ortho').T, norm='ortho')
            coefs[i:i+block, j:j+block] = np.round(B / q)
    return coefs, (h2, w2)

def block_dct_decode_channel(coefs, block=8, q=12):
    h, w = coefs.shape
    out = np.zeros((h,w), dtype=np.float32)
    for i in range(0, h, block):
        for j in range(0, w, block):
            C = coefs[i:i+block, j:j+block].astype(np.float32) * q
            blk = idct(idct(C.T, norm='ortho').T, norm='ortho')
            out[i:i+block, j:j+block] = blk
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------- AES-GCM helpers ----------
def derive_key(password, salt=b"improv_salt", length=32):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=length, salt=salt, iterations=100000)
    return kdf.derive(password.encode())

def aes_gcm_encrypt(data_bytes, password):
    key = derive_key(password)
    aesgcm = AESGCM(key)
    iv = os.urandom(12)
    ct = aesgcm.encrypt(iv, data_bytes, None)
    return iv + ct

def aes_gcm_decrypt(blob, password):
    key = derive_key(password)
    iv = blob[:12]; ct = blob[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(iv, ct, None)

# ---------- hyperchaos permute ----------
def derive_seeds(password):
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), b"hybrid_salt", 100000, dklen=32)
    a = int.from_bytes(key[:16], 'big') % (10**8)
    b = int.from_bytes(key[16:], 'big') % (10**8)
    x0 = 0.1357 + (a / 1e8) * 0.7
    y0 = 0.2468 + (b / 1e8) * 0.6
    return x0, y0

def hyperchaos_seq(length, x0, y0):
    x = x0; y = y0
    seq = np.empty(length, dtype=np.float64)
    for i in range(length):
        x = 3.9999 * x * (1 - x)
        y = 3.9876 * y * (1 - y)
        seq[i] = (x + 0.5*y) % 1.0
    return seq

def permute_bytes(data, password):
    arr = np.frombuffer(data, dtype=np.uint8)
    x0, y0 = derive_seeds(password)
    seq = hyperchaos_seq(len(arr), x0, y0)
    perm = np.argsort(seq)
    permuted = arr[perm]
    return permuted.tobytes(), perm

def unpermute_bytes(permuted_bytes, perm):
    arr = np.frombuffer(permuted_bytes, dtype=np.uint8)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    orig = arr[inv]
    return orig.tobytes()

# ---------- pipeline ----------
def encode_image(path, password):
    arr = img_to_channels(path)
    H, W, C = arr.shape
    channel_blobs = []
    shapes = []
    for ch in range(3):
        coefs, shape = block_dct_encode_channel(arr[:,:,ch], block=8, q=12)
        channel_blobs.append(coefs.tobytes())
        shapes.append(shape)
    # concat all channel bytes with headers (simple)
    header = f"{shapes[0][0]},{shapes[0][1]},{shapes[1][0]},{shapes[1][1]},{shapes[2][0]},{shapes[2][1]}|".encode()
    payload = header + b"||".join(channel_blobs)
    # AES-GCM encrypt payload
    ct = aes_gcm_encrypt(payload, password)
    # hyperchaos permute ciphertext bytes
    permuted, perm = permute_bytes(ct, password)
    return permuted, perm

def decode_image(permuted_bytes, perm, password):
    # unpermute
    ct = unpermute_bytes(permuted_bytes, perm)
    # AES decrypt
    payload = aes_gcm_decrypt(ct, password)
    # parse header
    header, data = payload.split(b"|", 1)
    parts = header.decode().split(",")
    # shapes and split data back to 3 channels
    # We used '||' delimiter
    blobs = data.split(b"||")
    channels = []
    for idx, blob in enumerate(blobs):
        h, w = int(parts[2*idx]), int(parts[2*idx+1])
        coefs = np.frombuffer(blob, dtype=np.int16).reshape((h,w))
        channel_img = block_dct_decode_channel(coefs, block=8, q=12)
        channels.append(channel_img)
    rgb = np.stack(channels, axis=2)
    return rgb

def demo(path):
    pwd = "StrongDemoPass!"
    print("Encoding (DCT+AES-GCM+hyperchaos)...")
    permuted, perm = encode_image(path, pwd)
    print("Encoded bytes:", len(permuted))
    print("Decoding back...")
    out = decode_image(permuted, perm, pwd)
    Image.fromarray(out).save("improved_restored.png")
    print("Saved improved_restored.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python improved_hybrid_aes_hyperchaos.py input_color.png")
        sys.exit(1)
    demo(sys.argv[1])

