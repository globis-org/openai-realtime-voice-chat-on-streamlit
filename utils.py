import av
import numpy as np


def hash_by_code(obj) -> int:
    """Hash function to detect code changes
    """
    import inspect
    return hash(inspect.getsource(obj))


def audio_frame_to_pcm_audio(frame: av.AudioFrame) -> bytes:
    return frame.to_ndarray().tobytes()


def pcm_audio_to_audio_frame(
    pcm_audio: bytes,
    *,
    format: str,
    layout: str,
    sample_rate: int
) -> av.AudioFrame:
    raw_data = np.frombuffer(pcm_audio, np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(raw_data, format = format, layout = layout)
    frame.sample_rate = sample_rate
    return frame


def get_blank_audio_frame(
    *,
    format: str,
    layout: str,
    samples: int,
    sample_rate: int
) -> av.AudioFrame:
    frame = av.AudioFrame(format = format, layout = layout, samples = samples)
    for p in frame.planes:
        p.update(bytes(p.buffer_size))
    frame.sample_rate = sample_rate
    return frame
