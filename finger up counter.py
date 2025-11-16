import cv2
from cvzone.HandTrackingModule import HandDetector
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.75)
ptime = time.time()

# Detect left or right hand using wrist & knuckle positions
def detect_hand_side(lm):
    # If landmark 17 is left of 5 → LEFT hand
    if lm[17][0] < lm[5][0]:
        return "Left"
    else:
        return "Right"

# Correct finger count
def count_fingers(hand):
    lm = hand["lmList"]
    side = detect_hand_side(lm)

    fingers = []

    # Thumb logic (different for L/R)
    if side == "Right":
        fingers.append(1 if lm[5][0] > lm[3][0] else 0)
    else:
        fingers.append(1 if lm[5][0] < lm[3][0] else 0)

    # Other 4 fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for t, p in zip(tips, pips):
        fingers.append(1 if lm[t][1] < lm[p][1] else 0)

    return fingers.count(1), side

# Beautiful display box
def drawBox(img, x, y, w, h, text, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.putText(img, text, (x + 15, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img, flipType=False)

    leftCount = 0
    rightCount = 0

    if hands:
        for hand in hands:
            cnt, side = count_fingers(hand)
            if side == "Left":
                leftCount = cnt
            else:
                rightCount = cnt

    total = leftCount + rightCount

    # UI Boxes (your old style)
    drawBox(img, 20, 20, 150, 70, f"Left: {leftCount}", (0, 150, 255))
    drawBox(img, 190, 20, 150, 70, f"Right: {rightCount}", (255, 120, 0))
    drawBox(img, 20, 110, 320, 70, f"Total: {total}", (0, 200, 0))

    # FPS text
    ctime = time.time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime
    cv2.putText(img, f"FPS: {fps}", (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
