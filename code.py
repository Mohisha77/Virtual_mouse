

import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from google.protobuf.json_format import MessageToDict

# Config
pyautogui.FAILSAFE = False
mp_draw = mp.solutions.drawing_utils
mp_handmod = mp.solutions.hands

# Gesture Definitions
class GestureCode(IntEnum):
    NONE = 0
    PINKY = 1
    RING = 2
    MIDDLE = 4
    LAST_THREE = 7
    INDEX = 8
    FIRST_TWO = 12
    LAST_FOUR = 15
    THUMB = 16
    OPEN_PALM = 31
    V_SIGN = 33
    TWO_FINGER_CLOSE = 34
    PINCH_MAIN = 35
    PINCH_SUB = 36

# Handedness
class HandSide(IntEnum):
    LEFT = 0
    RIGHT = 1

# Recognizer for a single hand
class HandGestureInterpreter:
    def __init__(self, label):
        self.finger_state = 0
        self.current_gesture = GestureCode.OPEN_PALM
        self.last_gesture = GestureCode.OPEN_PALM
        self.frame_stable_count = 0
        self.input_landmarks = None
        self.hand_side = label

    def update_landmarks(self, result):
        self.input_landmarks = result

    def _signed_distance(self, idx_pair):
        sign = -1 if self.input_landmarks.landmark[idx_pair[0]].y >= self.input_landmarks.landmark[idx_pair[1]].y else 1
        dx = self.input_landmarks.landmark[idx_pair[0]].x - self.input_landmarks.landmark[idx_pair[1]].x
        dy = self.input_landmarks.landmark[idx_pair[0]].y - self.input_landmarks.landmark[idx_pair[1]].y
        return math.sqrt(dx ** 2 + dy ** 2) * sign

    def _euclidean_dist(self, idx_pair):
        dx = self.input_landmarks.landmark[idx_pair[0]].x - self.input_landmarks.landmark[idx_pair[1]].x
        dy = self.input_landmarks.landmark[idx_pair[0]].y - self.input_landmarks.landmark[idx_pair[1]].y
        return math.sqrt(dx ** 2 + dy ** 2)

    def _depth_diff(self, idx_pair):
        return abs(self.input_landmarks.landmark[idx_pair[0]].z - self.input_landmarks.landmark[idx_pair[1]].z)

    def compute_finger_pattern(self):
        if not self.input_landmarks:
            return
        fingers = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger_state = 0
        for tip, mid, base in fingers:
            try:
                dist1 = self._signed_distance([tip, mid])
                dist2 = self._signed_distance([mid, base])
                ratio = round(dist1 / dist2, 1)
            except ZeroDivisionError:
                ratio = 0

            self.finger_state <<= 1
            if ratio > 0.5:
                self.finger_state |= 1

    def recognize(self):
        if not self.input_landmarks:
            return GestureCode.OPEN_PALM

        result_gesture = GestureCode.OPEN_PALM
        if self.finger_state in [GestureCode.LAST_THREE, GestureCode.LAST_FOUR] and self._euclidean_dist([8,4]) < 0.05:
            result_gesture = GestureCode.PINCH_SUB if self.hand_side == HandSide.LEFT else GestureCode.PINCH_MAIN

        elif self.finger_state == GestureCode.FIRST_TWO:
            dist = self._euclidean_dist([8,12])
            base_dist = self._euclidean_dist([5,9])
            ratio = dist / base_dist if base_dist != 0 else 0
            if ratio > 1.7:
                result_gesture = GestureCode.V_SIGN
            elif self._depth_diff([8,12]) < 0.1:
                result_gesture = GestureCode.TWO_FINGER_CLOSE
            else:
                result_gesture = GestureCode.MIDDLE
        else:
            result_gesture = self.finger_state

        if result_gesture == self.last_gesture:
            self.frame_stable_count += 1
        else:
            self.frame_stable_count = 0

        self.last_gesture = result_gesture

        if self.frame_stable_count > 4:
            self.current_gesture = result_gesture
        return self.current_gesture

# Handles actual system commands based on gestures
class SystemController:
    prev_cursor = None
    is_dragging = False
    pinch_started = False
    pinch_start_coords = (0, 0)
    pinch_delta = 0
    pinch_direction = None
    stability_count = 0
    last_level = 0
    threshold = 0.3

    @staticmethod
    def _calc_cursor_pos(landmark):
        screen_x, screen_y = pyautogui.size()
        hand_x = int(landmark[9].x * screen_x)
        hand_y = int(landmark[9].y * screen_y)
        if not SystemController.prev_cursor:
            SystemController.prev_cursor = (hand_x, hand_y)

        dx = hand_x - SystemController.prev_cursor[0]
        dy = hand_y - SystemController.prev_cursor[1]
        dist_sq = dx ** 2 + dy ** 2

        if dist_sq <= 25:
            ratio = 0
        elif dist_sq <= 900:
            ratio = 0.07 * math.sqrt(dist_sq)
        else:
            ratio = 2.1

        x_new = pyautogui.position()[0] + dx * ratio
        y_new = pyautogui.position()[1] + dy * ratio
        SystemController.prev_cursor = (hand_x, hand_y)
        return int(x_new), int(y_new)

    @staticmethod
    def _initiate_pinch(landmark):
        SystemController.pinch_start_coords = (landmark[8].x, landmark[8].y)
        SystemController.last_level = 0
        SystemController.pinch_delta = 0
        SystemController.stability_count = 0

    @staticmethod
    def _pinch_scroll(landmark):
        delta_x = (landmark[8].x - SystemController.pinch_start_coords[0]) * 10
        delta_y = (SystemController.pinch_start_coords[1] - landmark[8].y) * 10

        if abs(delta_y) > abs(delta_x) and abs(delta_y) > SystemController.threshold:
            direction = "vertical"
            level = round(delta_y, 1)
        elif abs(delta_x) > SystemController.threshold:
            direction = "horizontal"
            level = round(delta_x, 1)
        else:
            return

        if abs(level - SystemController.last_level) < SystemController.threshold:
            SystemController.stability_count += 1
        else:
            SystemController.stability_count = 0
            SystemController.last_level = level

        if SystemController.stability_count == 5:
            if direction == "vertical":
                pyautogui.scroll(120 if level > 0 else -120)
            elif direction == "horizontal":
                pyautogui.keyDown("ctrl")
                pyautogui.keyDown("shift")
                pyautogui.scroll(-120 if level > 0 else 120)
                pyautogui.keyUp("shift")
                pyautogui.keyUp("ctrl")
            SystemController.stability_count = 0

    @staticmethod
    def process_action(gesture, landmark):
        if gesture != GestureCode.OPEN_PALM:
            cursor = SystemController._calc_cursor_pos(landmark)
        else:
            SystemController.prev_cursor = None
            return

        if gesture == GestureCode.V_SIGN:
            pyautogui.moveTo(*cursor, duration=0.1)

        elif gesture == GestureCode.FIST:
            if not SystemController.is_dragging:
                SystemController.is_dragging = True
                pyautogui.mouseDown(button='left')
            pyautogui.moveTo(*cursor, duration=0.1)

        elif gesture in [GestureCode.MIDDLE, GestureCode.INDEX, GestureCode.TWO_FINGER_CLOSE]:
            pyautogui.click(button='left' if gesture == GestureCode.MIDDLE else 'right')
            SystemController.is_dragging = False

        elif gesture == GestureCode.PINCH_SUB:
            if not SystemController.pinch_started:
                SystemController._initiate_pinch(landmark)
                SystemController.pinch_started = True
            SystemController._pinch_scroll(landmark)
        else:
            SystemController.is_dragging = False
            SystemController.pinch_started = False

# Main Gesture Controller
class GestureMain:
    is_active = True
    capture = None
    frame_width = None
    frame_height = None
    primary_hand = None
    secondary_hand = None
    is_right_hand_dominant = True

    def __init__(self):
        GestureMain.capture = cv2.VideoCapture(0)
        GestureMain.frame_height = GestureMain.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureMain.frame_width = GestureMain.capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    @staticmethod
    def assign_hand_sides(results):
        l_hand = r_hand = None
        for idx, hand_info in enumerate(results.multi_handedness):
            label = MessageToDict(hand_info)['classification'][0]['label']
            if label == 'Right':
                r_hand = results.multi_hand_landmarks[idx]
            else:
                l_hand = results.multi_hand_landmarks[idx]

        if GestureMain.is_right_hand_dominant:
            GestureMain.primary_hand = r_hand
            GestureMain.secondary_hand = l_hand
        else:
            GestureMain.primary_hand = l_hand
            GestureMain.secondary_hand = r_hand

    def run(self):
        recognizer_primary = HandGestureInterpreter(HandSide.RIGHT)
        recognizer_secondary = HandGestureInterpreter(HandSide.LEFT)

        with mp_handmod.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while GestureMain.capture.isOpened() and GestureMain.is_active:
                success, frame = GestureMain.capture.read()
                if not success:
                    continue

                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                detection = hands.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if detection.multi_hand_landmarks:
                    GestureMain.assign_hand_sides(detection)
                    recognizer_primary.update_landmarks(GestureMain.primary_hand)
                    recognizer_secondary.update_landmarks(GestureMain.secondary_hand)

                    recognizer_primary.compute_finger_pattern()
                    recognizer_secondary.compute_finger_pattern()

                    detected_gesture = recognizer_secondary.recognize()
                    if detected_gesture == GestureCode.PINCH_SUB:
                        SystemController.process_action(detected_gesture, recognizer_secondary.input_landmarks)
                    else:
                        detected_gesture = recognizer_primary.recognize()
                        SystemController.process_action(detected_gesture, recognizer_primary.input_landmarks)

                    for hand_landmark in detection.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmark, mp_handmod.HAND_CONNECTIONS)
                else:
                    SystemController.prev_cursor = None

                cv2.imshow("Hand Gesture Control", frame)
                if cv2.waitKey(5) & 0xFF == 13:  # Enter key to quit
                    break

        GestureMain.capture.release()
        cv2.destroyAllWindows()

# Run if main
if __name__ == "__main__":
    gesture_ctrl = GestureMain()
    gesture_ctrl.run()
