"""Adapter from SIMPLE's joint-space reference expert to native SONIC."""

from contextlib import ExitStack
from pathlib import Path
import subprocess
import threading
import time

from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message
import numpy as np
import zmq

from simple.agents.wbc_debug_subscriber import WbcDebugSubscriber
from simple.core.action import ActionCmd

# The native planner wants upper-body positions in IsaacLab order. These
# indices select that order from the Unitree/MuJoCo motor order.
UPPER_BODY = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


def planner_packet(body, left, right, navigation, height, heading):
    velocity = np.asarray(navigation[:2], dtype=float)
    speed = float(np.linalg.norm(velocity))
    c, s = np.cos(heading), np.sin(heading)
    movement = np.array([[c, -s], [s, c]]) @ velocity / speed if speed else (0., 0.)
    yaw = navigation[3]
    return build_planner_message(
        1 if speed else 0, (*movement, 0.), (np.cos(yaw), np.sin(yaw), 0.),
        speed=speed or -1., height=height, upper_body_position=None if body is None else body[UPPER_BODY],
        left_hand_position=left, right_hand_position=right,
    )


class SonicReference:
    def __init__(self, robot, config, output):
        self.robot = robot
        self.config = config
        self.output = output

    def __enter__(self):
        stack = self.resources = ExitStack()
        try:
            deploy = Path('third_party/GR00T-WholeBodyControl/gear_sonic_deploy').resolve()
            context = stack.enter_context(zmq.Context())
            self.publisher = stack.enter_context(context.socket(zmq.PUB))
            self.publisher.setsockopt(zmq.LINGER, 0)
            port = self.publisher.bind_to_random_port('tcp://127.0.0.1')
            # Use the adjacent, conventional output port for this isolated pilot.
            debug_port = 5597
            self.debug = WbcDebugSubscriber(host='127.0.0.1', port=debug_port)
            stack.callback(self.debug.close)
            self.bridge = UnitreeSdk2Bridge(self.config)
            self.cache_state()

            log = stack.enter_context((self.output / 'sonic-controller.log').open('w'))
            cmd = [str(deploy / 'target/release/g1_deploy_onnx_ref'), 'lo',
                   'policy/release/model_decoder.onnx', 'reference/example/',
                   '--encoder-file', 'policy/release/model_encoder.onnx',
                   '--planner-file', 'planner/target_vel/V2/planner_sonic.onnx',
                   '--obs-config', 'policy/release/observation_config.yaml',
                   '--default-motion', 'squat_001__A359', '--input-type', 'zmq_manager',
                   '--zmq-host', '127.0.0.1', '--zmq-port', str(port),
                   '--zmq-out-port', str(debug_port), '--disable-crc-check',
                   '--logs-dir', str((self.output / 'sonic-controller').resolve())]
            # Headless control comes from ZMQ. Avoid treating inherited input as key presses.
            self.process = subprocess.Popen(cmd, cwd=deploy, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT)
            stack.callback(self.stop_controller)
            self.heartbeat_stop = threading.Event()

            def heartbeat():
                while not self.heartbeat_stop.is_set():
                    self.bridge.PublishLowState(self.state)
                    self.heartbeat_stop.wait(.005)

            self.heartbeat = threading.Thread(target=heartbeat, daemon=True)
            self.heartbeat.start()
            stack.callback(self.stop_heartbeat)
            return self
        except BaseException:
            stack.close()
            raise

    def __exit__(self, *exc):
        self.resources.close()

    def stop_controller(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def cache_state(self):
        self.state = {key: value.copy() if isinstance(value, np.ndarray) else value
                      for key, value in self.robot.prepare_obs().items()}

    def stop_heartbeat(self):
        self.heartbeat_stop.set()
        self.heartbeat.join()

    def publish_state(self):
        self.bridge.PublishLowState(self.robot.prepare_obs())

    def engage(self, heading, navigation_only=False):
        state = self.state
        body, left, right = (None, None, None) if navigation_only else (
            state['body_q'], state['left_hand_q'], state['right_hand_q'])
        hold = planner_packet(body, left, right, (0., 0., 0., heading),
                              -1. if navigation_only else .74, heading)
        start = build_command_message(start=True, stop=False, planner=True)
        deadline = time.monotonic() + 240
        planner_ready = False
        with (self.output / 'sonic-controller.log').open(errors='replace') as log:
            while self.process.poll() is None and time.monotonic() < deadline:
                self.publisher.send(start)
                self.publisher.send(hold)
                planner_ready |= 'motion name is planner_motion' in log.read()
                tick = self.debug.get_synced()
                if planner_ready and tick is not None and np.linalg.norm(tick['token']) > 0:
                    return
                time.sleep(.025)
        raise RuntimeError('SONIC did not engage. See sonic-controller.log.')

    def action(self, body, left, right, navigation, height, heading, split_action=None):
        if self.process.poll() is not None:
            raise RuntimeError('SONIC exited. See sonic-controller.log.')
        if split_action is not None:
            body = left = right = None
            height = -1.
        self.publisher.send(planner_packet(body, left, right, navigation, height, heading))
        action = split_action if split_action is not None else ActionCmd('wbc_torque')
        action.parameters['use_sensor'] = self.bridge.use_sensor
        self.refresh_action(action)
        return action

    def refresh_action(self, action):
        action.parameters['low_cmd'] = self.bridge.low_cmd
        if action.type == 'wbc_torque':
            action.parameters.update(left_hand_cmd=self.bridge.left_hand_cmd,
                                     right_hand_cmd=self.bridge.right_hand_cmd)
