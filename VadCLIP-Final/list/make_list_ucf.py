import os
import csv

root_path = 'D:/Tesi/features/VADClipLikeFeatures/'
txt = 'D:/Tesi/VadCLIP-main/list/Anomaly_Train.txt'

files = list(open(txt))
normal = []
count = 0

REP = 10  # ✅ come nel loro ucf_CLIP_rgbtest: usa solo __5

with open('CLIP_rgb.csv', 'w+', newline='') as f:  ## the name of feature list
    writer = csv.writer(f)
    writer.writerow(['path', 'label'])

    for file in files:
        file = file.strip()
        if not file:
            continue

        # --- versione originale (check su __0) ---
        # filename = root_path + file[:-5] + '__0.npy'

        # ✅ versione VadCLIP-like (test con solo __5)
        filename = root_path + file[:-4] + f'__{REP}.npy'   # ".mp4" -> "__5.npy"
        label = file.split('/')[0]

        if os.path.exists(filename):
            if 'Normal' in label:
                # --- ORIG: aggiunge 10 clip per i normal ---
                # filename0 = filename[:-5]   # toglie "0.npy" lasciando "...__"
                # for i in range(0, 10, 1):
                #     normal.append(filename0 + str(i) + '.npy')

                # ✅ NEW: per test aggiungi SOLO la clip rappresentativa (__5)
                normal.append(filename)

            else:
                # --- ORIG: scrive 10 clip per anomali ---
                # filename0 = filename[:-5]
                # for i in range(0, 10, 1):
                #     writer.writerow([filename0 + str(i) + '.npy', label])

                # ✅ NEW: per test scrivi SOLO la clip rappresentativa (__5)
                writer.writerow([filename, label])

        else:
            count += 1
            print(filename, flush=True)

    # --- ORIG: scrittura normal alla fine ---
    for file in normal:
        writer.writerow([file, 'Normal'])

print("missing:", count, flush=True)
