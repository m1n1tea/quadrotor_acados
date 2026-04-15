import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from .math_utils import (
    quaternion_inverse,
    skew_symmetric,
    transform_trajectory,
    v_dot_q,
)
from .quadrotor_model import QuadrotorParams


class Controller:
    def __init__(
        self,
        quad: QuadrotorParams,
        t_horizon: float = 1.0,
        n_nodes: int = 20,
        q_cost=None,
        r_cost=None,
        q_mask=None,
        rdrv_d_mat=None,
        model_name: str = "quad_3d_acados_mpc",
        solver_options=None,
        logger=None,
    ):
        if q_cost is None:
            q_cost = np.array(
                [10, 10, 10, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05]
            )
        if r_cost is None:
            r_cost = np.array([0.1, 0.1, 0.1, 0.1])

        self.T = t_horizon
        self.N = n_nodes
        self.quad = quad
        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        self.p = cs.MX.sym("p", 3)
        self.q = cs.MX.sym("a", 4)
        self.v = cs.MX.sym("v", 3)
        self.r = cs.MX.sym("r", 3)

        self.x = cs.vertcat(self.p, self.q, self.v, self.r)
        self.state_dim = 13

        u1 = cs.MX.sym("u1")
        u2 = cs.MX.sym("u2")
        u3 = cs.MX.sym("u3")
        u4 = cs.MX.sym("u4")
        self.u = cs.vertcat(u1, u2, u3, u4)

        self.quad_xdot_nominal = self.quad_dynamics(rdrv_d_mat)
        acados_models, nominal_with_gp = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u)["x_dot"], model_name
        )

        self.quad_xdot = {}
        for dyn_model_idx in nominal_with_gp.keys():
            dyn = nominal_with_gp[dyn_model_idx]
            self.quad_xdot[dyn_model_idx] = cs.Function(
                "x_dot", [self.x, self.u], [dyn], ["x", "u"], ["x_dot"]
            )

        q_diagonal = np.concatenate(
            (q_cost[:3], np.mean(q_cost[3:6])[np.newaxis], q_cost[3:])
        )
        if q_mask is not None:
            q_mask = np.concatenate((q_mask[:3], np.zeros(1), q_mask[3:]))
            q_diagonal *= q_mask

        for key_model in acados_models.values():
            nx = key_model.x.size()[0]
            nu = key_model.u.size()[0]
            ny = nx + nu
            n_param = key_model.p.size()[0] if isinstance(key_model.p, cs.MX) else 0

            ocp = AcadosOcp()
            ocp.model = key_model
            ocp.dims.N = self.N
            ocp.solver_options.tf = t_horizon

            ocp.dims.np = n_param
            ocp.parameter_values = np.zeros(n_param)

            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"

            ocp.cost.W = np.diag(np.concatenate((q_diagonal, r_cost)))
            ocp.cost.W_e = np.diag(q_diagonal)
            terminal_cost = (
                0
                if solver_options is None
                or not solver_options.get("terminal_cost", False)
                else 1
            )
            ocp.cost.W_e *= terminal_cost

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:nx, :nx] = np.eye(nx)
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vu[-4:, -4:] = np.eye(nu)
            ocp.cost.Vx_e = np.eye(nx)

            x_ref = np.zeros(nx)
            ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
            ocp.cost.yref_e = x_ref
            ocp.constraints.x0 = x_ref

            ocp.constraints.lbu = np.array([self.min_u] * 4)
            ocp.constraints.ubu = np.array([self.max_u] * 4)
            ocp.constraints.idxbu = np.array([0, 1, 2, 3])

            ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
            ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
            ocp.solver_options.integrator_type = "ERK"
            ocp.solver_options.print_level = 0
            ocp.solver_options.nlp_solver_type = (
                "SQP_RTI"
                if solver_options is None
                else solver_options.get("solver_type", "SQP_RTI")
            )

            # build_dir = Path("/tmp/quadrotor_acados")
            # build_dir.mkdir(parents=True, exist_ok=True)
            json_file = str(f"{key_model.name}_acados_ocp.json")
            self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_file)

        self.time_traj = None
        self.last_closest_index = 0

        self.logger = logger
        self.counter = 0

    def acados_setup_model(self, nominal, model_name):
        def fill_in_acados_model(x, u, p, dynamics, name):
            x_dot = cs.MX.sym("x_dot", dynamics.shape)
            f_impl = x_dot - dynamics

            model = AcadosModel()
            model.f_expl_expr = dynamics
            model.f_impl_expr = f_impl
            model.x = x
            model.xdot = x_dot
            model.u = u
            model.p = p
            model.name = name
            return model

        acados_models = {}
        dynamics_equations = {0: nominal}
        acados_models[0] = fill_in_acados_model(
            x=self.x, u=self.u, p=[], dynamics=nominal, name=model_name
        )
        return acados_models, dynamics_equations

    def quad_dynamics(self, rdrv_d):
        x_dot = cs.vertcat(
            self.p_dynamics(),
            self.q_dynamics(),
            self.v_dynamics(rdrv_d),
            self.w_dynamics(),
        )
        return cs.Function(
            "x_dot", [self.x[:13], self.u], [x_dot], ["x", "u"], ["x_dot"]
        )

    def p_dynamics(self):
        return self.v

    def q_dynamics(self):
        return 0.5 * cs.mtimes(skew_symmetric(self.r), self.q)

    def v_dynamics(self, rdrv_d):
        f_thrust = self.u * self.quad.max_thrust
        g = cs.vertcat(0.0, 0.0, -9.81)
        a_thrust = (
            cs.vertcat(
                0.0, 0.0, -(f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3])
            )
            / self.quad.mass
        )

        v_dyn = v_dot_q(a_thrust, self.q) + g

        if rdrv_d is not None:
            v_b = v_dot_q(self.v, quaternion_inverse(self.q))
            rdrv_drag = v_dot_q(cs.mtimes(rdrv_d, v_b), self.q)
            v_dyn += rdrv_drag

        return v_dyn

    def w_dynamics(self):
        f_thrust = self.u * self.quad.max_thrust
        y_f = cs.MX(self.quad.y_f)
        x_f = cs.MX(self.quad.x_f)
        c_f = cs.MX(self.quad.z_l_tau)

        return cs.vertcat(
            (
                cs.mtimes(f_thrust.T, y_f)
                + (self.quad.J[1] - self.quad.J[2]) * self.r[1] * self.r[2]
            )
            / self.quad.J[0],
            (
                -cs.mtimes(f_thrust.T, x_f)
                + (self.quad.J[2] - self.quad.J[0]) * self.r[2] * self.r[0]
            )
            / self.quad.J[1],
            (
                cs.mtimes(f_thrust.T, c_f)
                + (self.quad.J[0] - self.quad.J[1]) * self.r[0] * self.r[1]
            )
            / self.quad.J[2],
        )

    def update_trajectory(
        self, trajectory: np.ndarray, preferred_speed: float | None = None
    ):
        if preferred_speed is None:
            self.time_traj = trajectory
        else:
            self.time_traj = transform_trajectory(
                trajectory, preferred_speed * self.T / self.N
            )
        self.last_closest_index = 0

    def run_optimization(self, initial_state=None):
        if self.time_traj is None or len(self.time_traj) == 0:
            return np.zeros(4)

        if initial_state is None:
            initial_state = [0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0]

        x_init = np.stack(initial_state)

        self.acados_ocp_solver.set(0, "lbx", x_init)
        self.acados_ocp_solver.set(0, "ubx", x_init)

        starting_index = (
            np.argmin(
                np.sum(
                    (self.time_traj[self.last_closest_index :] - initial_state[:3])
                    ** 2,
                    axis=1,
                )
            )
            + self.last_closest_index
        )

        local_trajectory = self.time_traj[starting_index : starting_index + self.N + 1]
        if len(local_trajectory) < self.N + 1:
            local_trajectory = np.pad(
                local_trajectory,
                ((0, self.N + 1 - len(local_trajectory)), (0, 0)),
                "edge",
            )
        self.last_closest_index = starting_index

        if self.logger and self.counter % 100 == 0:
            self.logger.info(f"Current state={initial_state}")
            self.logger.info(f"Current trajectory={local_trajectory}")

        self.counter += 1

        for j in range(self.N):
            y_ref = np.array(
                [
                    local_trajectory[j, 0],
                    local_trajectory[j, 1],
                    local_trajectory[j, 2],
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            self.acados_ocp_solver.set(j, "yref", y_ref)

        y_refN = np.array(
            [
                local_trajectory[self.N, 0],
                local_trajectory[self.N, 1],
                local_trajectory[self.N, 2],
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        self.acados_ocp_solver.set(self.N, "yref", y_refN)

        self.acados_ocp_solver.solve()

        w_opt_acados = np.ndarray((self.N, 4))
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")

        return np.reshape(w_opt_acados, (-1))[:4]
