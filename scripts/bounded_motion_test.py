"""Shared input bounds for offline recorded-motion tests."""
import math

MAX_FRAMES = 180

def validate_sampling(frames, fps):
    if type(frames) is not int or not 2 <= frames <= MAX_FRAMES:
        raise ValueError('Require 2..180 source frames')
    if type(fps) is not int or not 1 <= fps <= 30:
        raise ValueError('Require integer review FPS in 1..30')

def minimum_body_frames(frames):
    validate_sampling(frames, 6)
    return math.ceil(0.8 * frames)

def validate_records(records, frames):
    validate_sampling(frames, 6)
    if not isinstance(records, list) or not 2 <= len(records) <= frames:
        raise ValueError('Require a bounded list of at least two motion packets')
    previous_id = previous_time = -1
    for record in records:
        frame_id, timestamp = record.get('frame_id'), record.get('timestamp_ms')
        if type(frame_id) is not int or not 0 <= frame_id < frames or frame_id <= previous_id:
            raise ValueError('Frame IDs must be unique, ordered and within the source timeline')
        if type(timestamp) is not int or not 0 <= timestamp <= 2**53-1 or timestamp <= previous_time:
            raise ValueError('Motion timestamps must be bounded and strictly increasing')
        previous_id, previous_time = frame_id, timestamp
