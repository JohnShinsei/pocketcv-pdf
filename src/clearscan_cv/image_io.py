from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DecodedImage:
    image: np.ndarray
    exif_orientation: int | None
    orientation_applied: bool
    decoded_size: dict[str, int]
    oriented_size: dict[str, int]

    def to_report(self) -> dict[str, object]:
        return {
            "exif_orientation": self.exif_orientation,
            "orientation_applied": self.orientation_applied,
            "decoded_size": self.decoded_size,
            "oriented_size": self.oriented_size,
        }


def _parse_tiff_orientation(payload: bytes) -> int | None:
    if len(payload) < 8:
        return None
    byte_order = payload[:2]
    if byte_order == b"II":
        endian = "little"
    elif byte_order == b"MM":
        endian = "big"
    else:
        return None
    if int.from_bytes(payload[2:4], endian) != 42:
        return None

    ifd_offset = int.from_bytes(payload[4:8], endian)
    if ifd_offset < 0 or ifd_offset + 2 > len(payload):
        return None
    entry_count = int.from_bytes(payload[ifd_offset : ifd_offset + 2], endian)
    entry_offset = ifd_offset + 2
    for index in range(entry_count):
        start = entry_offset + index * 12
        end = start + 12
        if end > len(payload):
            return None
        tag = int.from_bytes(payload[start : start + 2], endian)
        value_type = int.from_bytes(payload[start + 2 : start + 4], endian)
        value_count = int.from_bytes(payload[start + 4 : start + 8], endian)
        if tag == 0x0112 and value_type == 3 and value_count >= 1:
            value = int.from_bytes(payload[start + 8 : start + 10], endian)
            return value if 1 <= value <= 8 else None
    return None


def read_exif_orientation(data: bytes) -> int | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None

    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break

        marker = data[offset]
        offset += 1
        if marker in {0xD9, 0xDA}:
            break
        if offset + 2 > len(data):
            break

        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2:
            break
        segment_start = offset + 2
        segment_end = offset + segment_length
        if segment_end > len(data):
            break

        if marker == 0xE1 and data[segment_start : segment_start + 6] == b"Exif\x00\x00":
            return _parse_tiff_orientation(data[segment_start + 6 : segment_end])
        offset = segment_end
    return None


def apply_exif_orientation(image: np.ndarray, orientation: int | None) -> tuple[np.ndarray, bool]:
    if orientation in {None, 1}:
        return image, False
    if orientation == 2:
        return cv2.flip(image, 1), True
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180), True
    if orientation == 4:
        return cv2.flip(image, 0), True
    if orientation == 5:
        return cv2.transpose(image), True
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), True
    if orientation == 7:
        return cv2.flip(cv2.transpose(image), -1), True
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), True
    return image, False


def decode_image_bytes(data: bytes, flags: int = cv2.IMREAD_COLOR) -> DecodedImage:
    array = np.frombuffer(data, dtype=np.uint8)
    ignore_orientation = getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)
    decode_flags = flags if flags == cv2.IMREAD_UNCHANGED else flags | ignore_orientation
    image = cv2.imdecode(array, decode_flags)
    if image is None:
        raise ValueError("Unsupported or unreadable image.")

    decoded_size = {"width": int(image.shape[1]), "height": int(image.shape[0])}
    orientation = read_exif_orientation(data)
    oriented, applied = apply_exif_orientation(image, orientation)
    oriented_size = {"width": int(oriented.shape[1]), "height": int(oriented.shape[0])}
    return DecodedImage(
        image=oriented,
        exif_orientation=orientation,
        orientation_applied=applied,
        decoded_size=decoded_size,
        oriented_size=oriented_size,
    )


def decode_image_file(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> DecodedImage:
    return decode_image_bytes(Path(path).read_bytes(), flags=flags)
