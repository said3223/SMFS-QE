"""Cortex Bus Controller implementation.

This module provides a Python implementation inspired by the CORTEX BUS CONTROLLER
architecture diagram.  The controller orchestrates signal routing, state
buffering, and trace monitoring between several subsystems: Guardian Cortex,
LUET reasoning, reflective goal inputs, environment inputs, and the SMFS-QE
stream.

The design embraces a message-passing style.  Each component exposes a narrow
interface, which keeps the flow of data easy to test in isolation while still
making composition straightforward.  The controller is intentionally kept
framework-agnostic: it does not depend on any specific event loop or async
runtime, and can be embedded inside larger systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# Signal primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Signal:
    """Base signal travelling through the Cortex Bus.

    Attributes
    ----------
    channel:
        Human-readable channel name (e.g. "reflective", "input").
    source:
        Identifier for the producer (e.g. "guardian", "agent").
    payload:
        Opaque data that downstream consumers can interpret.
    timestamp:
        Time the signal was created.  Defaults to ``datetime.utcnow`` if
        omitted by the caller.
    metadata:
        Optional dictionary for additional annotations.
    """

    channel: str
    source: str
    payload: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceEvent:
    """Represents a monitoring event emitted by the TraceMonitoring unit."""

    origin: str
    description: str
    payload: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Core components corresponding to the diagram
# ---------------------------------------------------------------------------

class SignalRouting:
    """Routes signals into the Cortex Bus pipeline.

    The router supports registering predicates paired with handlers.  When a
    signal arrives, the router evaluates predicates in order and invokes the
    first matching handler.  This makes it easy to funnel different kinds of
    messages towards dedicated processing branches.
    """

    def __init__(self) -> None:
        self._routes: List[Tuple[Callable[[Signal], bool], Callable[[Signal], None]]] = []

    def register_route(
        self,
        predicate: Callable[[Signal], bool],
        handler: Callable[[Signal], None],
    ) -> None:
        self._routes.append((predicate, handler))

    def route(self, signal: Signal) -> None:
        for predicate, handler in self._routes:
            if predicate(signal):
                handler(signal)
                return
        # Fall back to a default handler that simply ignores the signal.
        # In production environments this could log or raise for visibility.


class StateBuffer:
    """Maintains a rolling buffer of Cortex state snapshots."""

    def __init__(self, *, maxlen: int = 128) -> None:
        self._snapshots: Deque[Dict[str, Any]] = deque(maxlen=maxlen)

    def update(self, state: Dict[str, Any]) -> None:
        # Store a copy to prevent downstream mutation from affecting history.
        self._snapshots.append(dict(state))

    def latest(self) -> Optional[Dict[str, Any]]:
        return self._snapshots[-1] if self._snapshots else None

    def snapshot_history(self) -> List[Dict[str, Any]]:
        return list(self._snapshots)


class TraceMonitoring:
    """Observes activity within the Cortex Bus and emits trace events."""

    def __init__(self) -> None:
        self._events: List[TraceEvent] = []
        self._subscribers: List[Callable[[TraceEvent], None]] = []

    def record(self, origin: str, description: str, payload: Any) -> TraceEvent:
        event = TraceEvent(origin=origin, description=description, payload=payload)
        self._events.append(event)
        for subscriber in self._subscribers:
            subscriber(event)
        return event

    def subscribe(self, consumer: Callable[[TraceEvent], None]) -> None:
        self._subscribers.append(consumer)

    def history(self) -> List[TraceEvent]:
        return list(self._events)


class SMFSStream:
    """Simple stream that forwards TraceEvents to an SMFS-QE consumer."""

    def __init__(self, consumer: Optional[Callable[[TraceEvent], None]] = None) -> None:
        self._consumer = consumer

    def push(self, event: TraceEvent) -> None:
        if self._consumer is not None:
            self._consumer(event)


# ---------------------------------------------------------------------------
# Cortex Bus Controller façade
# ---------------------------------------------------------------------------

class CortexBusController:
    """High-level façade combining routing, buffering, and monitoring."""

    def __init__(
        self,
        *,
        state_buffer_size: int = 128,
        smfs_consumer: Optional[Callable[[TraceEvent], None]] = None,
    ) -> None:
        self.signal_routing = SignalRouting()
        self.state_buffer = StateBuffer(maxlen=state_buffer_size)
        self.trace_monitoring = TraceMonitoring()
        self.smfs_stream = SMFSStream(smfs_consumer)
        self._guardian_state: Dict[str, Any] = {}

        # Wire default routing behaviour according to the diagram.
        self._setup_default_routes()

        # By default the SMFS stream subscribes to all trace events.
        self.trace_monitoring.subscribe(self.smfs_stream.push)

    # -- Public API -----------------------------------------------------

    def ingest_reflective_goal(self, payload: Any, **metadata: Any) -> None:
        self._dispatch_signal(
            Signal(channel="reflective", source="goal", payload=payload, metadata=metadata)
        )

    def ingest_input(self, payload: Any, **metadata: Any) -> None:
        self._dispatch_signal(
            Signal(channel="input", source="environment", payload=payload, metadata=metadata)
        )

    def guardian_reason(self, insight: Any, **metadata: Any) -> None:
        """Guardian Cortex reasoning signal (arrow 1 in the diagram)."""
        self._dispatch_signal(
            Signal(channel="guardian", source="guardian_cortex", payload=insight, metadata=metadata)
        )

    def guardian_trace(self, observation: Any, **metadata: Any) -> None:
        """Guardian Cortex trace signal (arrow 2 in the diagram)."""
        event = self.trace_monitoring.record(
            origin="guardian_cortex", description="guardian-trace", payload=observation
        )
        self.smfs_stream.push(event)

    def luet_reasoning(self, inference: Any, **metadata: Any) -> None:
        self._dispatch_signal(
            Signal(channel="luet", source="luet_reasoning", payload=inference, metadata=metadata)
        )

    def latest_state(self) -> Optional[Dict[str, Any]]:
        return self.state_buffer.latest()

    def trace_history(self) -> List[TraceEvent]:
        return self.trace_monitoring.history()

    # -- Internal helpers ----------------------------------------------

    def _setup_default_routes(self) -> None:
        def _store_guardian(signal: Signal) -> None:
            self._guardian_state = {
                "type": "guardian_reasoning",
                "payload": signal.payload,
                "metadata": signal.metadata,
                "timestamp": signal.timestamp,
            }
            self._buffer_state(reasoning=self._guardian_state)

        def _buffer_reflective(signal: Signal) -> None:
            state = {
                "type": "reflective_goal",
                "payload": signal.payload,
                "metadata": signal.metadata,
                "timestamp": signal.timestamp,
            }
            self._buffer_state(reflective_goal=state)

        def _buffer_input(signal: Signal) -> None:
            state = {
                "type": "input_signal",
                "payload": signal.payload,
                "metadata": signal.metadata,
                "timestamp": signal.timestamp,
            }
            self._buffer_state(latest_input=state)

        def _buffer_luet(signal: Signal) -> None:
            state = {
                "type": "luet_reasoning",
                "payload": signal.payload,
                "metadata": signal.metadata,
                "timestamp": signal.timestamp,
            }
            self._buffer_state(luet_inference=state)

        self.signal_routing.register_route(
            lambda s: s.channel == "guardian", _store_guardian
        )
        self.signal_routing.register_route(
            lambda s: s.channel == "reflective", _buffer_reflective
        )
        self.signal_routing.register_route(lambda s: s.channel == "input", _buffer_input)
        self.signal_routing.register_route(lambda s: s.channel == "luet", _buffer_luet)

    def _dispatch_signal(self, signal: Signal) -> None:
        self.trace_monitoring.record(
            origin="signal-routing",
            description=f"received-{signal.channel}",
            payload={
                "source": signal.source,
                "payload": signal.payload,
                "metadata": signal.metadata,
                "timestamp": signal.timestamp,
            },
        )
        self.signal_routing.route(signal)

    def _buffer_state(self, **updates: Dict[str, Any]) -> None:
        combined_state = dict(self._guardian_state)
        if self.state_buffer.latest():
            combined_state.update(self.state_buffer.latest())
        combined_state.update(updates)
        self.state_buffer.update(combined_state)
        self.trace_monitoring.record(
            origin="state-buffer",
            description="state-updated",
            payload=combined_state,
        )


# ---------------------------------------------------------------------------
# Example Guardian Cortex stub (optional helper for integrators)
# ---------------------------------------------------------------------------

class GuardianCortex:
    """Minimal helper that produces reasoning and trace signals."""

    def __init__(self, controller: CortexBusController) -> None:
        self._controller = controller

    def reason(self, insight: Any, **metadata: Any) -> None:
        self._controller.guardian_reason(insight, **metadata)

    def trace(self, observation: Any, **metadata: Any) -> None:
        self._controller.guardian_trace(observation, **metadata)


class LUETReasoningModule:
    """Placeholder component encapsulating LUET reasoning output."""

    def __init__(self, controller: CortexBusController) -> None:
        self._controller = controller

    def emit(self, inference: Any, **metadata: Any) -> None:
        self._controller.luet_reasoning(inference, **metadata)
