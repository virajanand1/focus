import time
GRACE_PERIOD = 2.0

class FocusTracker:
    def __init__(self):
        self._status = "FOCUSED"
        self.distracted_time = 0.0
        self.distracted_count = 0
        self._pending_status = None
        self._pending_since = None
        self._last_tick = time.time()
    
    def update(self, raw_status):
        dt = time.time() - self._last_tick
        self._last_tick = time.time()
        if self._status == "LOOKING_AWAY":
            self.distracted_time += dt 
        
        raw_status = "LOOKING_AWAY" if raw_status == "NO_FACE" else raw_status

        if raw_status == self._status:
            self._pending_status = None
            self._pending_since = None
            return

        if raw_status != self._pending_status:
            self._pending_status = raw_status
            self._pending_since = time.time()

        if self._pending_since and time.time() - self._pending_since > GRACE_PERIOD:
            self._status = self._pending_status
            if self._status == "LOOKING_AWAY":
                self.distracted_count += 1
            self._pending_status = None
            self._pending_since = None

    def current_state(self):
        return self._status
    
    def reset(self):
        self.__init__()