"""Fetch the author's MSMT17-trained OSNet checkpoint with pinned SHA256."""
import hashlib
from pathlib import Path
import urllib.request

URL = 'https://drive.usercontent.google.com/download?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x&export=download&confirm=t'
SHA256 = '48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727'
DESTINATION = Path(__file__).resolve().parents[1] / 'models' / 'osnet_x1_0_msmt17.pth'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    if DESTINATION.is_file() and digest(DESTINATION) == SHA256:
        print(f'Already verified: {DESTINATION}')
        return
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix('.download')
    try:
        with urllib.request.urlopen(URL, timeout=45) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if digest(temporary) != SHA256:
            raise RuntimeError('ReID model checksum mismatch; refusing to install')
        temporary.replace(DESTINATION)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Verified: {DESTINATION}')


if __name__ == '__main__':
    main()
