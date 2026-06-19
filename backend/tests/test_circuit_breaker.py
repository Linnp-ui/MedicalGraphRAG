"""Unit tests for circuit_breaker module."""

import time
import pytest
from unittest.mock import patch

from src.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
    get_circuit_breaker,
    reset_all_circuits,
)
import src.core.circuit_breaker as cb_module


# ---------------------------------------------------------------------------
# TestCircuitBreakerConfig
# ---------------------------------------------------------------------------

class TestCircuitBreakerConfig:
    def test_defaults(self):
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 20
        assert config.recovery_timeout == 60
        assert config.half_open_max_calls == 10

    def test_custom_values(self):
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30,
            half_open_max_calls=3,
        )
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30
        assert config.half_open_max_calls == 3


# ---------------------------------------------------------------------------
# TestCircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_record_success_resets_failure_count(self):
        cb = CircuitBreaker("test")
        cb._failure_count = 5
        cb.record_success()
        assert cb._failure_count == 0

    def test_record_failure_increments(self):
        cb = CircuitBreaker("test")
        cb.record_failure()
        assert cb._failure_count == 1
        cb.record_failure()
        assert cb._failure_count == 2

    def test_transitions_to_open_after_threshold(self):
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    @patch("src.core.circuit_breaker.time.time")
    def test_half_open_transition_after_timeout(self, mock_time):
        mock_time.return_value = 100.0
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=1)
        cb = CircuitBreaker("test", config)

        # Trip the breaker to OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Still within timeout -> stays OPEN
        mock_time.return_value = 100.5
        assert cb.is_open() is True

        # After timeout -> transitions to HALF_OPEN
        mock_time.return_value = 101.5
        assert cb.is_open() is False
        assert cb.state == CircuitState.HALF_OPEN

    @patch("src.core.circuit_breaker.time.time")
    def test_recovery_from_half_open(self, mock_time):
        mock_time.return_value = 100.0
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=1, half_open_max_calls=3)
        cb = CircuitBreaker("test", config)

        # Trip to OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Advance past recovery timeout -> HALF_OPEN
        mock_time.return_value = 102.0
        cb.is_open()  # triggers transition

        # Successful calls in HALF_OPEN
        for _ in range(3):
            cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    @patch("src.core.circuit_breaker.time.time")
    def test_half_open_failure_reopens(self, mock_time):
        mock_time.return_value = 100.0
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=1)
        cb = CircuitBreaker("test", config)

        # Trip to OPEN
        cb.record_failure()

        # Advance past recovery timeout -> HALF_OPEN
        mock_time.return_value = 102.0
        cb.is_open()  # triggers transition
        assert cb.state == CircuitState.HALF_OPEN

        # Failure in HALF_OPEN -> back to OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_call_success(self):
        cb = CircuitBreaker("test")
        result = cb.call(lambda: 42)
        assert result == 42

    def test_call_open_raises_error(self):
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        cb.record_failure()  # trips to OPEN
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: 42)

    def test_call_open_returns_fallback(self):
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        cb.record_failure()  # trips to OPEN
        result = cb.call(lambda: 42, fallback="fallback_value")
        assert result == "fallback_value"

    def test_is_open_after_failures(self):
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        assert cb.is_open() is False
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open() is True


# ---------------------------------------------------------------------------
# TestCircuitBreakerGlobals
# ---------------------------------------------------------------------------

class TestCircuitBreakerGlobals:
    def setup_method(self):
        # Reset global circuit breaker registry between tests
        cb_module._circuit_breakers.clear()

    def test_get_circuit_breaker_returns_same(self):
        cb1 = get_circuit_breaker("my_service")
        cb2 = get_circuit_breaker("my_service")
        assert cb1 is cb2

    def test_reset_all_circuits(self):
        config = CircuitBreakerConfig(failure_threshold=1)
        cb1 = get_circuit_breaker("svc_a", config)
        cb2 = get_circuit_breaker("svc_b", config)

        cb1.record_failure()
        cb2.record_failure()
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        reset_all_circuits()
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED
        assert cb1._failure_count == 0
        assert cb2._failure_count == 0
