"""Validate and decode uploads in memory, without retaining user images."""

from io import BytesIO
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIME_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
MAX_PIXELS = 20_000_000


def decode_image(content: bytes, content_type: str) -> Image.Image:
    """Verify format and structure, apply EXIF orientation, then detach RGB pixels."""
    if content_type not in MIME_FORMATS:
        raise ValueError("Use a JPEG, PNG or WEBP image.")
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a nonempty image no larger than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                if source.format != MIME_FORMATS[content_type]:
                    raise ValueError("The decoded image format does not match its content type.")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 20 megapixel processing limit.")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Use a still image; animated images are not supported.")
                source.verify()
            with Image.open(BytesIO(content)) as source:
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("The image is corrupt, unsupported or too large to decode.") from exc
