"""PMOCO route objectives on top of the unchanged MVMoE environments."""

import torch


def route_objectives(depot_node_xy, selected_node_list, open_route=False, loc_scaler=None):
    """Return [total distance, longest route] for each [batch, POMO] tour.

    Incoming edges are assigned to the route ending at their destination depot.
    This avoids the depot-prefix boundary error in the original PMOCO code.
    A final depot is appended so a closed route is completed even when its
    representation omits the final depot; an open route's closing edge is zero.
    """
    batch_size, pomo_size, _ = selected_node_list.shape
    closing_depot = selected_node_list.new_zeros(batch_size, pomo_size, 1)
    selected = torch.cat((selected_node_list, closing_depot), dim=2)
    all_xy = depot_node_xy[:, None].expand(-1, pomo_size, -1, -1)
    ordered = all_xy.gather(2, selected[..., None].expand(-1, -1, -1, 2))
    start = depot_node_xy[:, :1, :][:, None].expand(-1, pomo_size, -1, -1)
    previous = torch.cat((start, ordered[:, :, :-1]), dim=2)
    lengths = (ordered - previous).norm(p=2, dim=-1)
    if loc_scaler:
        lengths = torch.round(lengths * loc_scaler) / loc_scaler
    if open_route:
        lengths = lengths * (selected != 0)

    # The closing depot belongs to the preceding route, not the next one.
    depot = (selected == 0).long()
    route_index = depot.cumsum(dim=2) - depot
    per_route = lengths.new_zeros(batch_size, pomo_size, selected.size(2) + 1)
    per_route.scatter_add_(2, route_index, lengths)
    return torch.stack((lengths.sum(dim=2), per_route.max(dim=2).values), dim=2)


def validate_routes(selected_node_list, problem_size):
    """Check complete customer coverage without inventing objective points."""
    counts = selected_node_list.new_zeros(*selected_node_list.shape[:2], problem_size + 1)
    counts.scatter_add_(2, selected_node_list, torch.ones_like(selected_node_list))
    return (counts[:, :, 1:] == 1).all(dim=2)


class MultiObjectiveEnv:
    """Delegate constraints to one original environment and track route costs."""

    def __init__(self, env, check_feasibility=False):
        self.env = env
        self.open_route = env.problem.startswith('O')
        self.check_feasibility = check_feasibility
        self.original_reward = None
        self.objectives = None
        self.total_distance = None
        self.route_distance = None
        self.max_route_distance = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _add_objective_state(self, state):
        state.total_distance = self.total_distance
        state.max_route_distance = self.max_route_distance
        return state

    def reset(self):
        state, reward, done = self.env.reset()
        self.total_distance = torch.zeros_like(self.env.load)
        self.route_distance = torch.zeros_like(self.env.load)
        self.max_route_distance = torch.zeros_like(self.env.load)
        self.original_reward = None
        self.objectives = None
        self._add_objective_state(self.env.step_state)
        return self._add_objective_state(state), reward, done

    def pre_step(self):
        state, reward, done = self.env.pre_step()
        return self._add_objective_state(state), reward, done

    def step(self, selected):
        if self.check_feasibility:
            chosen_mask = self.env.ninf_mask.gather(2, selected[:, :, None]).squeeze(2)
            if not torch.isfinite(chosen_mask).all():
                raise RuntimeError('Selected an infeasible action in {}'.format(self.env.problem))

        previous_coord = self.env.current_coord
        already_finished = self.env.finished.clone()
        state, reward, done = self.env.step(selected)
        edge = (self.env.current_coord - previous_coord).norm(p=2, dim=-1)
        if self.env.loc_scaler:
            edge = torch.round(edge * self.env.loc_scaler) / self.env.loc_scaler
        if self.open_route:
            edge = edge * (selected != 0)
        edge = edge * (~already_finished)

        self.total_distance = self.total_distance + edge
        self.route_distance = self.route_distance + edge
        self.max_route_distance = torch.maximum(self.max_route_distance, self.route_distance)
        self.route_distance = self.route_distance.masked_fill(selected == 0, 0)
        self._add_objective_state(state)
        if done:
            self.original_reward = reward
            self.objectives = torch.stack((self.total_distance, self.max_route_distance), dim=2)
            reward = -self.objectives
            if self.check_feasibility:
                if not validate_routes(self.env.selected_node_list, self.env.problem_size).all():
                    raise RuntimeError('Incomplete/repeated customer in {}'.format(self.env.problem))
                torch.testing.assert_close(self.objectives[:, :, 0], -self.original_reward,
                                           rtol=1e-5, atol=1e-5)
                parsed = route_objectives(self.env.depot_node_xy, self.env.selected_node_list,
                                          self.open_route, self.env.loc_scaler)
                torch.testing.assert_close(self.objectives, parsed, rtol=1e-5, atol=1e-5)
        return state, reward, done
