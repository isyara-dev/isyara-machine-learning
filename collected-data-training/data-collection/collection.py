import os
import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from datetime import datetime

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Buat header untuk CSV
def create_csv_header():
    header = ['label']
    for hand in ['left', 'right']:
        for i in range(21):
            header.extend([f'{hand}_{i}_x', f'{hand}_{i}_y', f'{hand}_{i}_z'])
    return header

# Ekstrak landmark sebagai flat list (1D)
def extract_landmarks(results):
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if handedness.classification[0].label == 'Left':
                left_hand = coords
            else:
                right_hand = coords
    return left_hand, right_hand

# Ubah flat list menjadi list 3D [[x, y, z], ...]
def reshape_landmarks(flat_landmarks):
    return [[flat_landmarks[i], flat_landmarks[i+1], flat_landmarks[i+2]] for i in range(0, len(flat_landmarks), 3)]

# Normalisasi landmark berdasarkan posisi pergelangan tangan (wrist)
def normalize_landmarks(landmarks, wrist):
    return [[lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]] for lm in landmarks]

# Simpan ke file CSV
def save_to_csv(filename, data, header):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)

# Fungsi utama
def main():
    label = input("Masukkan label gesture (contoh: A, B, C): ").strip()
    while True:
        try:
            num_samples = int(input("Masukkan jumlah sampel yang ingin diambil: "))
            break
        except ValueError:
            print("Masukkan angka yang valid.")

    while True:
        mode_input = input("Pilih mode pengambilan data:\n1. Satu tangan\n2. Dua tangan\nMasukkan pilihan (1/2): ").strip()
        if mode_input in ['1', '2']:
            mode = int(mode_input)
            break
        else:
            print("Masukkan pilihan yang valid (1 atau 2).")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka kamera.")
        return

    print("Tekan 's' untuk mulai, 'q' untuk keluar.")

    header = create_csv_header()
    one_hand_file = 'one_hand_landmarks.csv'
    two_hand_file = 'two_hand_landmarks.csv'

    collecting = False
    sample_count = 0
    start_time = None
    preparing = False
    prep_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Label: {label} | Sampel: {sample_count}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Landmark Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Keluar dari program.")
            break
        elif key == ord('s') and not collecting and not preparing:
            print("Persiapkan posisi tangan. Mulai dalam 5 detik...")
            preparing = True
            prep_start_time = time.time()
        
        if preparing:
            elapsed_prep = time.time() - prep_start_time
            cv2.putText(frame, f"Bersiap... {int(5 - elapsed_prep)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if elapsed_prep >= 5:
                collecting = True
                preparing = False
                start_time = time.time()
                sample_count = 0
                print("Pengambilan data dimulai...")

        if collecting and sample_count < num_samples:
            current_time = time.time()
            if current_time - start_time >= 1:  # Ambil setiap 1 detik
                if mode == 1:
                    if results.multi_hand_landmarks:
                        # Ambil satu tangan saja (tangan pertama)
                        landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
                        wrist = landmarks[0]
                        normalized = normalize_landmarks(landmarks, wrist)

                        left_hand = [0.0] * 63
                        right_hand = [coord for lm in normalized for coord in lm]
                        num_hands_detected = 1
                    else:
                        left_hand = [0.0] * 63
                        right_hand = [0.0] * 63
                        num_hands_detected = 0

                elif mode == 2:
                    left_hand, right_hand = extract_landmarks(results)
                    num_hands_detected = 0

                    if any(coord != 0.0 for coord in left_hand):
                        reshaped_left = reshape_landmarks(left_hand)
                        left_hand = np.array(normalize_landmarks(reshaped_left, reshaped_left[0])).flatten().tolist()
                        num_hands_detected += 1

                    if any(coord != 0.0 for coord in right_hand):
                        reshaped_right = reshape_landmarks(right_hand)
                        right_hand = np.array(normalize_landmarks(reshaped_right, reshaped_right[0])).flatten().tolist()
                        num_hands_detected += 1

                if (mode == 1 and num_hands_detected == 1) or (mode == 2 and num_hands_detected == 2):
                    data_row = [label] + left_hand + right_hand
                    filename = one_hand_file if mode == 1 else two_hand_file
                    save_to_csv(filename, data_row, header)
                    sample_count += 1
                    print(f"Sampel {sample_count}/{num_samples} disimpan.")
                else:
                    print("Jumlah tangan tidak sesuai mode.")

                start_time = current_time

        if collecting and sample_count >= num_samples:
            print("Pengambilan data selesai.")
            collecting = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
