# Code for communicating with the MCio mod
from dataclasses import dataclass, asdict, field
from typing import Set, List

import cbor2
import zmq

DEFAULT_HOST = "localhost"
DEFAULT_ACTION_PORT = 5556
DEFAULT_STATE_PORT = 5557
DEFAULT_ACTION_ADDR = f"tcp://{DEFAULT_HOST}:{DEFAULT_ACTION_PORT}"
DEFAULT_STATE_ADDR = f"tcp://{DEFAULT_HOST}:{DEFAULT_STATE_PORT}"

# State packets received from MCio
@dataclass
class StatePacket:
    seq: int = 0
    frame_png: bytes = b""
    health: float = 0.0
    message: str = ""
    inventory_main: List = field(default_factory=list)
    inventory_armor: List = field(default_factory=list)
    inventory_offhand: List = field(default_factory=list)

    @classmethod
    def unpack(cls, data: bytes) -> 'StatePacket':
        try:
            decoded_dict = cbor2.loads(data)
        except Exception as e:
            print(f"CBOR load error: {type(e).__name__}: {e}")
            return None

        try:
            rv = cls(**decoded_dict)
        except Exception as e:
            # This means the received packet doesn't match StatePacket
            print(f"StatePacket decode error: {type(e).__name__}: {e}")
            return None

        return rv


# Action packets sent by the agent to MCio
@dataclass
class ActionPacket:
    seq: int = 0           # sequence number
    keys_pressed: Set[int] = field(default_factory=set)
    keys_released: Set[int] = field(default_factory=set)
    mouse_buttons_pressed: Set[int] = field(default_factory=set)
    mouse_buttons_released: Set[int] = field(default_factory=set)
    mouse_pos_update: bool = False
    mouse_pos_x: int = 0
    mouse_pos_y: int = 0
    key_reset: bool = False
    message: str = ""

    def pack(self) -> bytes:
        return cbor2.dumps(asdict(self))
    

# Connections to MCio mod
class Connection:
    def __init__(self, action_addr=DEFAULT_ACTION_ADDR, state_addr=DEFAULT_STATE_ADDR):
        # Initialize ZMQ context
        self.zmq_context = zmq.Context()

        # Socket to send commands
        self.action_socket = self.zmq_context.socket(zmq.PUB)
        self.action_socket.bind(action_addr)
        
        # Socket to receive state updates
        self.state_socket = self.zmq_context.socket(zmq.SUB)
        self.state_socket.connect(state_addr)
        self.state_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def send_action(self, action:ActionPacket):
        self.action_socket.send(action.pack())

    def recv_state(self) -> StatePacket | None:
        try:
            pbytes = self.state_socket.recv()
        except zmq.error.ContextTerminated:
            return None
        
        # This may also return None if there was an unpack error.
        # XXX Maybe these errors should be separated. A context error can happen during shutdown.
        # We could continue after a parse error.
        return StatePacket.unpack(pbytes)

    # TODO add a simplified interface that encapsulates threads

    def close(self):
        self.action_socket.close()
        self.state_socket.close()
        self.zmq_context.term()

