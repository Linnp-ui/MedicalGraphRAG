import time
import threading
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 20
    recovery_timeout: int = 60
    half_open_max_calls: int = 10


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def _try_half_open(self) -> bool:
        """Check if circuit should transition from OPEN to HALF_OPEN.
        Returns True if transition occurred."""
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and self._should_attempt_reset()
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                return True
            return False

    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.config.recovery_timeout

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit {self.name} CLOSED after successful recovery")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name} OPEN after half-open failure")
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name} OPEN after {self._failure_count} failures")

    def is_open(self) -> bool:
        """Check if circuit is open, attempting half-open transition."""
        self._try_half_open()
        with self._lock:
            return self._state == CircuitState.OPEN

    def call(self, func: Callable[[], Any], fallback: Any = None) -> Any:
        self._try_half_open()

        with self._lock:
            state = self._state

        if state == CircuitState.OPEN:
            logger.warning(f"Circuit {self.name} is OPEN, using fallback")
            if fallback is not None:
                return fallback
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")

        try:
            result = func()
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class CircuitOpenError(Exception):
    pass


_circuit_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    with _breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def reset_all_circuits() -> None:
    with _breakers_lock:
        for breaker in _circuit_breakers.values():
            breaker._state = CircuitState.CLOSED
            breaker._failure_count = 0
    logger.info("All circuit breakers reset")


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitOpenError",
    "get_circuit_breaker",
    "reset_all_circuits",
]
