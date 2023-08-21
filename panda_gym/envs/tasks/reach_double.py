from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import TaskDouble
from panda_gym.utils import distance


class ReachDouble(TaskDouble):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
        goal_random=True,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.goal_random=goal_random
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target1",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="target2",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal1, self.goal2 = self._sample_goal()
        self.sim.set_base_pose("target1", self.goal1, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.goal2, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        if self.goal_random:
            goal1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
            goal2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        else:
            goal1 = np.array([-0.10652074, 0.00213265, 0.19745056])
            goal2 = np.array([0.12452769, 0.04585412, 0.11220955])
        return goal1, goal2

    def is_success(self, achieved_goal: np.ndarray, desired_goal1: np.ndarray, desired_goal2: np.ndarray) -> np.ndarray:
        d1 = distance(achieved_goal, desired_goal1)
        d2 = distance(achieved_goal, desired_goal2)
        return np.array(d1 < self.distance_threshold or d2 < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal1, desired_goal2, info: Dict[str, Any]) -> np.ndarray:
        d1 = distance(achieved_goal, desired_goal1)
        d2 = distance(achieved_goal, desired_goal2)
        if self.reward_type == "sparse":
            return -np.array(d1 > self.distance_threshold or d2 > self.distance_threshold, dtype=np.float32)
        else:
            return -(2.0*d1 + d2).astype(np.float32)