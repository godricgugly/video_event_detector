from src.detection.event_detector import EventDetector


def test_event_not_triggered_immediately():
    detector = EventDetector(fps=10, threshold=0.8, duration_sec=0.7)

    # feed high similarity but not enough frames yet
    for _ in range(3):  # less than 7 frames required (10 * 0.7)
        assert detector.update(0.9) is False

def test_event_triggers_after_sustained_similarity():
    detector = EventDetector(fps=10, threshold=0.8, duration_sec=0.5)

    # 5 frames needed (10 * 0.5)

    triggered = False
    for _ in range(10):
        if detector.update(0.9):
            triggered = True

    assert triggered is True

def test_event_turns_off_on_drop():
    detector = EventDetector(fps=10, threshold=0.8, duration_sec=0.5, cooldown_sec=0)

    # trigger event
    for _ in range(10):
        detector.update(0.9)

    assert detector.active is True

    # drop similarity
    for _ in range(10):
        state = detector.update(0.1)

    assert state is False
    assert detector.active is False