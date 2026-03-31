import time
GRACE_PERIOD = 2.0

class FocusTracker:
    def __init__(self):
        self._state = "FOCUSED"
        self.distracted_time = 0.0
        self.distracted_count = 0
        self._pending_state = None
        self._pending_since = None
        self._last_tick = time.time()
    
    def update(self, raw_status):
        dt = time.time() - self._last_tick
        self._last_tick = time.time()
        if self._state == "LOOKING_AWAY":
            self.distracted_time += dt 
        
        raw_status = "LOOKING_AWAY" if raw_status == "NO_FACE" else raw_status

        if raw_status != self._pending_state:
            self._pending_state = raw_status
            self._pending_since = time.time()

        elif raw_status == self._state:
            self._pending_state = None
            self._pending_since = None

        if time.time() - self._pending_since > GRACE_PERIOD:
            self._state = self._pending_state
            if self._state == "LOOKING_AWAY":
                self.distracted_count += 1
            self._pending_state = None
            self._pending_since = None

    
    def reset(self):
        self.__init__()