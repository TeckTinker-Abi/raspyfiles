import sounddevice as sd
import numpy as np
import librosa
import joblib
import RPi.GPIO as GPIO
from spafe.features.gfcc import gfcc

# ===============================
# GPIO SETUP
# ===============================
LED_PIN = 18
BUZZER_PIN = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

GPIO.output(LED_PIN, GPIO.LOW)
GPIO.output(BUZZER_PIN, GPIO.LOW)

# ===============================
# LOAD LOCKED MODELS
# ===============================
cry_model = joblib.load("models/cry_detection_model.pkl")
reason_model = joblib.load("models/cry_reason_final_3class_model.pkl")
reason_scaler = joblib.load("models/cry_reason_final_3class_scaler.pkl")

reason_labels = {
    0: "Hunger",
    1: "Discomfort",
    2: "Sleepiness"
}

# ===============================
# AUDIO SETTINGS
# ===============================
SR = 16000
WINDOW_SEC = 2
HOP_SEC = 1

WINDOW_SAMPLES = SR * WINDOW_SEC
HOP_SAMPLES = SR * HOP_SEC

CRY_THRESHOLD = 0.6
CONFIRM_WINDOWS = 3

cry_counter = 0
audio_buffer = np.array([])

print("🎧 Real-time Cry Detection Started (Ctrl+C to stop)")

# ===============================
# AUDIO CALLBACK
# ===============================
def audio_callback(indata, frames, time, status):
    global audio_buffer, cry_counter

    audio_buffer = np.concatenate((audio_buffer, indata[:, 0]))

    if len(audio_buffer) < WINDOW_SAMPLES:
        return

    window = audio_buffer[:WINDOW_SAMPLES]
    audio_buffer = audio_buffer[HOP_SAMPLES:]

    # ---------- Cry Detection (13 GFCC) ----------
    gfcc_feat = gfcc(
        window,
        fs=SR,
        num_ceps=13,
        nfilts=26,
        nfft=512
    )

    feat_13 = np.mean(gfcc_feat, axis=0).reshape(1, -1)
    cry_prob = cry_model.predict_proba(feat_13)[0][1]
    cry_pred = cry_model.predict(feat_13)[0]

    if cry_pred == 1 and cry_prob >= CRY_THRESHOLD:
        cry_counter += 1
    else:
        cry_counter = 0
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

    # ---------- Confirmed Cry ----------
    if cry_counter >= CONFIRM_WINDOWS:
        cry_counter = 0

        mean_feat = np.mean(gfcc_feat, axis=0)
        std_feat = np.std(gfcc_feat, axis=0)
        feat_26 = np.concatenate([mean_feat, std_feat]).reshape(1, -1)

        feat_26 = reason_scaler.transform(feat_26)
        reason_pred = reason_model.predict(feat_26)[0]
        reason_conf = reason_model.predict_proba(feat_26).max()

        # GPIO ALERT
        GPIO.output(LED_PIN, GPIO.HIGH)
        GPIO.output(BUZZER_PIN, GPIO.HIGH)

        print("\n🚨 CRY DETECTED")
        print("Reason:", reason_labels[reason_pred])
        print("Confidence:", round(reason_conf, 2))
        print("----------------------")

# ===============================
# START STREAM
# ===============================
try:
    with sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=HOP_SAMPLES,
        callback=audio_callback
    ):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    GPIO.cleanup()
