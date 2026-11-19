"""Adapter from SIMPLE's joint-space reference expert to native SONIC."""

from contextlib import ExitStack
import os
from pathlib import Path
import struct
import subprocess
import threading
import time

from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message
import numpy as np
import zmq

from simple.agents.wbc_debug_subscriber import WbcDebugSubscriber
from simple.agents.lockstep import LockstepBarrier
from simple.agents.lockstep_unitree_bridge import LockstepUnitreeBridge
from simple.core.action import ActionCmd

# The native planner wants upper-body positions in IsaacLab order. These
# indices select that order from the Unitree/MuJoCo motor order.
UPPER_BODY = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


def movement_goal(navigation, heading):
    velocity = np.asarray(navigation[:2], dtype=float)
    speed = float(np.linalg.norm(velocity))
    c, s = np.cos(heading), np.sin(heading)
    movement = np.array([[c, -s], [s, c]]) @ velocity / speed if speed else (0., 0.)
    yaw = navigation[3]
    return int(speed > 0), (*movement, 0.), (np.cos(yaw), np.sin(yaw), 0.), speed or -1.


class PlannerBridge(LockstepUnitreeBridge):
    """Carry the navigation goal in the same simulated LowState as its boundary."""

    def set_navigation(self, navigation, heading, height):
        mode, movement, facing, speed = movement_goal(navigation, heading)
        # The patched loopback controller reads NAV1 from the 40 remote bytes.
        self.low_state.wireless_remote[:] = struct.pack('<4sI8f', b'NAV1', mode, *movement, *facing, speed, height)


def planner_packet(body, left, right, navigation, height, heading):
    mode, movement, facing, speed = movement_goal(navigation, heading)
    return build_planner_message(
        mode, movement, facing, speed=speed, height=height, upper_body_position=body[UPPER_BODY],
        left_hand_position=left, right_hand_position=right,
    )


class SonicReference:
    def __init__(self, robot, config, output, *, lockstep=False):
        self.robot = robot
        self.config = config
        self.output = output
        self.lockstep = lockstep

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
            binary = deploy / 'target/release/g1_deploy_onnx_ref'
            controller_env = os.environ.copy()
            if self.lockstep:
                binary = (Path(os.environ.get('SONIC_TOOLS_DIR', '.tools/sonic'))
                          / 'g1_deploy_planner_lockstep').resolve()
                self.bridge = PlannerBridge(self.config)
                self.barrier = LockstepBarrier(self.bridge, None, physics_dt=self.config['SIMULATE_DT'])
                stack.callback(self.barrier.close)
                controller_env['SONIC_FORCE_UNITREE_DDS'] = '1'
            else:
                self.debug = WbcDebugSubscriber(host='127.0.0.1', port=debug_port)
                stack.callback(self.debug.close)
                self.bridge = UnitreeSdk2Bridge(self.config)
                self.cache_state()

            log = stack.enter_context((self.output / 'sonic-controller.log').open('w'))
            cmd = [str(binary), 'lo',
                   'policy/release/model_decoder.onnx', 'reference/example/',
                   '--encoder-file', 'policy/release/model_encoder.onnx',
                   '--planner-file', 'planner/target_vel/V2/planner_sonic.onnx',
                   '--obs-config', 'policy/release/observation_config.yaml',
                   '--default-motion', 'squat_001__A359', '--input-type', 'zmq_manager',
                   '--zmq-host', '127.0.0.1', '--zmq-port', str(port),
                   '--zmq-out-port', str(debug_port), '--disable-crc-check',
                   '--logs-dir', str((self.output / 'sonic-controller').resolve())]
            if self.lockstep:
                cmd.append('--sim-lockstep')
            # Headless control comes from ZMQ. Avoid treating inherited input as key presses.
            self.process = subprocess.Popen(cmd, cwd=deploy, env=controller_env, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT)
            stack.callback(self.stop_controller)
            if self.lockstep:
                self.wait_for_lockstep()
                return self
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

    def wait_for_lockstep(self):
        deadline = time.monotonic() + 240
        with (self.output / 'sonic-controller.log').open(errors='replace') as log:
            while self.process.poll() is None and time.monotonic() < deadline:
                if 'G1Deploy object created successfully' in log.read():
                    state = self.robot.prepare_obs()
                    self.barrier.begin_session(state, mj_time=state['time'])
                    return
                time.sleep(.05)
        raise RuntimeError('SONIC lockstep controller did not become ready. See sonic-controller.log.')

    def cache_substep(self):
        state = self.robot.prepare_obs()
        self.barrier.cache_substep(state, mj_time=state['time'])

    def engage(self, heading):
        state = self.state
        hold = planner_packet(state['body_q'], state['left_hand_q'], state['right_hand_q'],
                              (0., 0., 0., heading), .74, heading)
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
        if self.lockstep:
            self.bridge.set_navigation(navigation, heading, -1.)
            snapshot = self.barrier.commit_boundary(None)
            split_action.parameters.update(low_cmd=snapshot.low_cmd, use_sensor=self.bridge.use_sensor)
            return split_action
        self.publisher.send(planner_packet(body, left, right, navigation, height, heading))
        action = ActionCmd('wbc_torque')
        action.parameters['use_sensor'] = self.bridge.use_sensor
        self.refresh_action(action)
        return action

    def refresh_action(self, action):
        action.parameters['low_cmd'] = self.bridge.low_cmd
        if action.type == 'wbc_torque':
            action.parameters.update(left_hand_cmd=self.bridge.left_hand_cmd,
                                     right_hand_cmd=self.bridge.right_hand_cmd)
