import os
import json
import cv2
import numpy as np
import mediapipe as mp

# =========================
# CONFIG
# =========================
INPUT_ROOT = "videos"
OUTPUT_ROOT = "landmarks_out"

# Optional speed-up:
# 1 = process every frame (max quality)
# 2 = process 1 frame out of 2 (2x faster)
# 3 = 3x faster, etc.
PROCESS_EVERY_N_FRAMES = 1

# MediaPipe Face Mesh expects 468 landmarks
EXPECTED_LANDMARKS = 468

mp_face_mesh = mp.solutions.face_mesh


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rel_output_paths(video_path: str):
    """
    Replicates the INPUT_ROOT folder structure inside OUTPUT_ROOT.
    Example:
      videos/Folder1/Part1/01/0.mov
    becomes:
      landmarks_out/Folder1/Part1/01/0_landmarks.npz
      landmarks_out/Folder1/Part1/01/0_meta.json
    """
    rel = os.path.relpath(video_path, INPUT_ROOT)
    rel_dir = os.path.dirname(rel)
    base = os.path.splitext(os.path.basename(rel))[0]

    out_dir = os.path.join(OUTPUT_ROOT, rel_dir)
    ensure_dir(out_dir)

    out_npz = os.path.join(out_dir, f"{base}_landmarks.npz")
    out_meta = os.path.join(out_dir, f"{base}_meta.json")
    return out_npz, out_meta


def safe_landmark_array(lm_list):
    """
    Convert MediaPipe landmarks list -> numpy array (468,3).
    If shape is unexpected, pad/crop to (468,3) to avoid np.stack errors.
    """
    arr = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)  # (N,3)

    if arr.shape == (EXPECTED_LANDMARKS, 3):
        return arr

    # Fix unexpected shape by padding/cropping
    fixed = np.full((EXPECTED_LANDMARKS, 3), np.nan, dtype=np.float32)
    m = min(arr.shape[0], EXPECTED_LANDMARKS)
    fixed[:m, :] = arr[:m, :]
    return fixed


def extract_video_landmarks(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # can be 0 for some codecs

    landmarks_list = []
    frame_indices = []

    face_detected_frames = 0
    processed_frames = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # Optional frame skipping
            if PROCESS_EVERY_N_FRAMES > 1 and (frame_id % PROCESS_EVERY_N_FRAMES != 0):
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                arr = safe_landmark_array(lm)
                face_detected_frames += 1
            else:
                arr = np.full((EXPECTED_LANDMARKS, 3), np.nan, dtype=np.float32)

            landmarks_list.append(arr)
            frame_indices.append(frame_id)
            processed_frames += 1

            # progress ping
            if processed_frames % 300 == 0:
                print(f"  processed {processed_frames} frames...")

    cap.release()

    # Stack safely (now all items have same shape)
    if landmarks_list:
        landmarks = np.stack(landmarks_list, axis=0).astype(np.float32)  # (T,468,3)
    else:
        landmarks = np.empty((0, EXPECTED_LANDMARKS, 3), dtype=np.float32)

    frame_indices = np.array(frame_indices, dtype=np.int32)

    meta = {
        "video_path": video_path,
        "fps": float(fps) if fps is not None else None,
        "total_frames_est": total_frames_est,
        "processed_frames": int(landmarks.shape[0]),
        "process_every_n_frames": PROCESS_EVERY_N_FRAMES,
        "face_detected_frames": int(face_detected_frames),
        "face_detected_ratio": float(face_detected_frames / max(1, landmarks.shape[0])),
        "landmarks_shape": list(landmarks.shape),  # [T,468,3]
        "coords": "mediapipe_normalized_xyz",
        "note": "Frames with no detected face contain NaN landmarks."
    }

    return landmarks, frame_indices, meta


def main():
    ensure_dir(OUTPUT_ROOT)

    exts = (".mov", ".mp4", ".avi", ".mkv")

    for root, _, files in os.walk(INPUT_ROOT):
        for f in files:
            if not f.lower().endswith(exts):
                continue

            video_path = os.path.join(root, f)
            out_npz, out_meta = rel_output_paths(video_path)

            # Skip if already extracted (prevents redoing everything)
            if os.path.exists(out_npz) and os.path.exists(out_meta):
                print(f"SKIP (exists): {video_path}")
                continue

            print(f"\nExtracting: {video_path}")
            landmarks, frame_indices, meta = extract_video_landmarks(video_path)

            # Save NPZ
            np.savez_compressed(
                out_npz,
                landmarks=landmarks,
                frame_indices=frame_indices
            )

            # Save meta JSON
            with open(out_meta, "w", encoding="utf-8") as fp:
                json.dump(meta, fp, indent=2)

            print(f"Saved: {out_npz}")
            print(f"Saved: {out_meta}")


if __name__ == "__main__":
    main()
