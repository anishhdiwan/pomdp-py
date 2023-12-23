"""
This is an implementation of the AC-POMCP algorithm. The algorithm builds on the PO-UCT algorithm which is its base class

This implementation is based on the implementations of other POMDP algorithms from https://h2r.github.io/pomdp-py/html/installation.html
"""

from pomdp_py.framework.basics cimport Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel
from pomdp_py.framework.planner cimport Planner
from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.representations.belief.particles cimport particle_reinvigoration
from pomdp_py.algorithms.po_uct cimport VNode, RootVNode, QNode, POUCT, RandomRollout
import copy
import time
import random
import math
import torch


cdef class VNodeParticles(VNode):
    """AC_POMCP's VNode maintains particle belief"""
    def __init__(self, num_visits, belief=Particles([])):
        self.num_visits = num_visits
        self.belief = belief
        self.children = {}  # a -> QNode
    def __str__(self):
        return "VNode(%.3f, %.3f, %d | %s)" % (self.num_visits, self.value, len(self.belief),
                                               str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

cdef class RootVNodeParticles(RootVNode):
    def __init__(self, num_visits, history, belief=Particles([])):
        # vnodeobj = VNodeParticles(num_visits, value, belief=belief)
        RootVNode.__init__(self, num_visits, history)
        self.belief = belief
    @classmethod
    def from_vnode(cls, vnode, history):
        rootnode = RootVNodeParticles(vnode.num_visits, history, belief=vnode.belief)
        rootnode.children = vnode.children
        return rootnode

cdef class AC_POMCP(POUCT):

    """AC_POMCP description ..."""

    def __init__(self,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 rollout_policy=RandomRollout(), action_prior=None,
                 show_progress=False, pbar_update_interval=5):
        super().__init__(max_depth=max_depth,
                         planning_time=planning_time,
                         num_sims=num_sims,
                         discount_factor=discount_factor,
                         exploration_const=exploration_const,
                         num_visits_init=num_visits_init,
                         value_init=value_init,
                         rollout_policy=rollout_policy,
                         action_prior=action_prior,
                         show_progress=show_progress,
                         pbar_update_interval=pbar_update_interval)
        print("THIS IS THE AC_POMCP ALGO!")
        temp = torch.tensor([1., 2.])
        print(temp)

    @property
    def update_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return True

    def plan(self, agent):
        # Only works if the agent's belief is particles
        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented in particles.\n"\
                            "AC_POMCP not usable. Please convert it to particles.")
        return POUCT.plan(self, agent)

    cpdef update(self, Agent agent, Action real_action, Observation real_observation,
                 state_transform_func=None):
        """
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.

        `state_transform_func`: Used to add artificial transform to states during
            particle reinvigoration. Signature: s -> s_transformed
        """
        if not isinstance(agent.belief, Particles):
            raise TypeError("agent's belief is not represented in particles.\n"\
                            "AC_POMCP not usable. Please convert it to particles.")
        if not hasattr(agent, "tree"):
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if agent.tree[real_action][real_observation] is None:
            # Never anticipated the real_observation. No reinvigoration can happen.
            raise ValueError("Particle deprivation.")
        # Update the tree; Reinvigorate the tree's belief and use it
        # as the updated belief for the agent.
        agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation],
                                                   agent.history)
        tree_belief = agent.tree.belief
        agent.set_belief(particle_reinvigoration(tree_belief,
                                                 len(agent.init_belief.particles),
                                                 state_transform_func=state_transform_func))
        # If observation was never encountered in simulation, then tree will be None;
        # particle reinvigoration will occur.
        if agent.tree is not None:
            agent.tree.belief = copy.deepcopy(agent.belief)

    cpdef _simulate(AC_POMCP self,
                    State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth):
        total_reward = POUCT._simulate(self, state, history, root, parent, observation, depth)
        if depth == 1 and root is not None:
            root.belief.add(state)  # belief update happens as simulation goes.
        return total_reward

    def _VNode(self, agent=None, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        if root:
            # agent cannot be None.
            return RootVNodeParticles(self._num_visits_init,
                                      agent.history,
                                      belief=copy.deepcopy(agent.belief))
        else:
            if agent is None:
                return VNodeParticles(self._num_visits_init,
                                      belief=Particles([]))
            else:
                return VNodeParticles(self._num_visits_init,
                                      belief=copy.deepcopy(agent.belief))
