# universal_hybrid_image_processor.py
# DCT + AES-GCM + Hyperchaos, automatically handles grayscale or RGB
# Generates Encrypted, Decrypted, Reconstructed images

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
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
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

# ---------- hyperchaos ----------
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
def encode_image(arr, password):
    H, W, C = arr.shape
    channel_blobs = []
    shapes = []
    for ch in range(C):
        coefs, shape = block_dct_encode_channel(arr[:,:,ch], block=8, q=12)
        channel_blobs.append(coefs.tobytes())
        shapes.append(shape)
    header = ",".join([f"{h},{w}" for h,w in shapes]).encode() + b"|"
    payload = header + b"||".join(channel_blobs)
    ct = aes_gcm_encrypt(payload, password)
    permuted, perm = permute_bytes(ct, password)
    return permuted, perm

def decode_image(permuted_bytes, perm, password):
    ct = unpermute_bytes(permuted_bytes, perm)
    payload = aes_gcm_decrypt(ct, password)
    header, data = payload.split(b"|",1)
    parts = list(map(int, header.decode().split(",")))
    blobs = data.split(b"||")
    channels = []
    for idx, blob in enumerate(blobs):
        h, w = parts[2*idx], parts[2*idx+1]
        coefs = np.frombuffer(blob, dtype=np.int16).reshape((h,w))
        channels.append(block_dct_decode_channel(coefs, block=8, q=12))
    rgb = np.stack(channels, axis=2)
    return rgb

def process_image(path, password="StrongDemoPass!"):
    img = Image.open(path)
    is_gray = img.mode in ["L", "1"]  # simple detection
    arr = img_to_channels(path)
    permuted, perm = encode_image(arr, password)
    
    # Save encrypted as binary
    enc_name = os.path.splitext(os.path.basename(path))[0] + "_encrypted.bin"
    with open(enc_name, "wb") as f:
        f.write(permuted)
    
    # Decrypt & reconstruct
    decrypted_arr = decode_image(permuted, perm, password)
    
    dec_name = os.path.splitext(os.path.basename(path))[0] + "_decrypted.png"
    Image.fromarray(decrypted_arr).save(dec_name)
    
    rec_name = os.path.splitext(os.path.basename(path))[0] + "_reconstructed.png"
    Image.fromarray(decrypted_arr).save(rec_name)
    
    print(f"Processed '{path}' | Encrypted: {enc_name} | Decrypted: {dec_name} | Reconstructed: {rec_name}")

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python universal_hybrid_image_processor.py input_image.png")
        sys.exit(1)
    process_image(sys.argv[1])
