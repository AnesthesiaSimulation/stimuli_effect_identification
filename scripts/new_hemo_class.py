from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
import casadi as cas
from scipy.signal import TransferFunction, lfilter


def fsig(x, c50, gam): return x**gam / (c50**gam + x**gam)  # quick definition of sigmoidal function


class NewHemoModel:
    r"""This class implements the mechanically based model of Haemodynamics proposed in [Su2023]_.

    Particularly, Haemodynamics are considered to be a dynamical system with the following dynamics:

    .. math::

        \dot{TPR} = k_{in\_TPR} RMAP^{FB} (1+EFF_{prop\_TPR}) - k_{out} TPR (1 - EFF_{remi\_TPR})

    .. math::

        \dot{SV^*} = k_{in\_SV} RMAP^{FB} (1+EFF_{prop\_SV}) - k_{out} SV^* (1 - EFF_{remi\_SV})

    .. math::

        \dot{HR^*} = k_{in\_HR} RMAP^{FB} - k_{out} HR^* (1 - EFF_{remi\_HR})

    Where TPR stands for total peripheral resistance, SV for stroke volume, and HR for heart rate.
    The different effects are sigmoid functions depending on propofol and remifentanil concentrations.

    Moreover, we consider the effect of norepinephrine on the haemodynamics as a simple addition of a sigmoid curve
    to MAP from the models of [Beloeil2005]_ and [Oualha2014]_. To model the implications of norepinephrine, we use:

    .. math::

        MAP\_wanted(t) = MAP_{no\_nore}(t) + norepinephrine\_effect(t)

    where :math:`MAP_{no\_nore}(t)` is the MAP computed without norepinephrine, and
    :math:`norepinephrine\_effect(t)` is the effect of norepinephrine on MAP modeled by a sigmoid function.
    The dynamics of TPR including norepinephrine is then:

    .. math::

        \begin{align}
        \dot{TPR}(t) = & \frac{k_{in\_TPR}}{RMAP(t)^{FB}}(1 + EFF_{prop\_TPR}(t)) \\
        & - k_{out} TPR(t) (1 - EFF_{remi\_TPR}(t)) \\
        & + k_{nore}(MAP_{wanted}(t)- MAP(t))
        \end{align}

    Parameters
    ----------
    age : float
        Age of the patient in years.
    ts : float
        Sampling time in seconds.
    model : str, optional
        Model to use, only 'Su' is available. The default is 'Su'.
    nore_model : str, optional
        Model to use for norepinephrine, 'Beloeil' and 'Oualha' are available. The default is 'Beloeil'.
    random : bool, optional
        Add uncertainties in the parameters. The default is False.
    hr_base : float, optional
        Baseline heart rate (bpm). The default is None, which will use the value from the Su model.
    sv_base : float, optional
        Baseline stroke volume (mL). The default is None, which will use the value from the Su model.
    map_base : float, optional
        Baseline mean arterial pressure (mmHg). The default is None, which will use the value from the Su model.

    References
    ----------
    .. [Su2023] H. Su, J. V. Koomen, D. J. Eleveld, M. M. R. F. Struys, and P. J. Colin,
            “Pharmacodynamic mechanism-based interaction model for the haemodynamic effects of remifentanil
            and propofol in healthy volunteers,” *British Journal of Anaesthesia*, vol. 131, no. 2,
            pp. 222–233, Aug. 2023. doi:10.1016/j.bja.2023.04.043.

    .. [Beloeil2005] H. Beloeil, J.-X. Mazoit, D. Benhamou, and J. Duranteau,
            “Norepinephrine kinetics and dynamics in septic shock and trauma patients,”
            *BJA: British Journal of Anaesthesia*, vol. 95, no. 6, pp. 782–788, Dec. 2005.
            doi:10.1093/bja/aei259.

    .. [Oualha2014] M. Oualha et al., “Population pharmacokinetics and haemodynamic effects of norepinephrine
            in hypotensive critically ill children,” *British Journal of Clinical Pharmacology*,
            vol. 78, no. 4, pp. 886–897, 2014. doi:10.1111/bcp.12412.
    """

    def __init__(self,
                 age: float,
                 ts: float,
                 model: str = 'Su',
                 nore_model: str = 'Beloeil',
                 random: bool = False,
                 hr_base: float = None,
                 sv_base: float = None,
                 map_base: float = None,
                 ):
        """
        Initialize the class.

        Returns
        -------
        None.

        """
        self.ts = ts

        if model == 'Su':
            self.sv_base = 82.2
            self.hr_base = 56.1
            self.tpr_base = 0.0163
            self.k_out = 0.072 / 60  # (1/s)
            self.k_in_tpr = self.k_out * self.tpr_base
            self.k_in_sv = self.k_out * self.sv_base
            self.k_in_hr = self.k_out * self.hr_base
            self.fb = -0.661
            self.hr_sv = 0.312
            self.k_ltde = 0.067 / 60  # (1/s)
            self.ltde_sv = 0.0899
            self.ltde_hr = 0.121
            self.c50_propo_tpr = 3.21  # (µg/ml)
            self.emax_propo_tpr = -0.778  # (%)
            self.gamma_propo_tpr = 1.83
            self.c50_propo_sv = 0.44  # (µg/ml)
            emax_propo_sv_type = -0.154
            self.emax_propo_sv = emax_propo_sv_type * np.exp(0.0333 * (age - 35))
            self.age_max_sv = 0.033
            self.c50_remi_tpr = 4.59  # (ng/ml)
            self.emax_remi_tpr = -1
            self.sl_remi_hr = 0.0327  # (ng/ml)
            self.sl_remi_sv = 0.0581  # (ng/ml)
            self.int_hr = -0.119  # (ng/ml)
            self.c50_int_hr = 0.20  # (µg/ml)
            self.int_tpr = 1
            self.int_sv = -0.212  # (ng/ml)

            if hr_base is not None:
                self.hr_base = hr_base / (1 + self.ltde_hr)
                self.sv_base = sv_base / (1 + self.ltde_sv)
                self.tpr_base = map_base / (hr_base * sv_base)
                self.k_in_tpr = self.k_out * self.tpr_base
                self.k_in_sv = self.k_out * self.sv_base
                self.k_in_hr = self.k_out * self.hr_base

            # uncertainties values
            self.w_block1_mu = [0, 0, 0]
            self.w_block1_cov = [
                [0.0328, -0.0244, 0],
                [-0.0244, 0.0528, -0.0233],
                [0, -0.0233, 0.0242]
            ]

            self.w_block2_mu = [0, 0]
            self.w_block2_cov = [[0.00382, 0.00329], [0.00329, 0.00868]]

            self.w_c50_propo_tpr = np.sqrt(0.44)
            self.w_emax_remi_tpr = np.sqrt(0.449)
        else:
            raise ValueError("only Su is implemented as model")
        if nore_model == 'Beloeil':
            # see H. Beloeil, J.-X. Mazoit, D. Benhamou, and J. Duranteau, “Norepinephrine kinetics and dynamics
            # in septic shock and trauma patients,” BJA: British Journal of Anaesthesia,
            # vol. 95, no. 6, pp. 782–788, Dec. 2005, doi: 10.1093/bja/aei259.
            self.emax_nore_map = 98.7
            self.c50_nore_map = 7.04
            self.gamma_nore_map = 1.8
            w_emax_nore_map = 0
            w_c50_nore_map = 1.64
            w_gamma_nore_map = 0
        elif nore_model == 'Oualha':
            # see M. Oualha et al., “Population pharmacokinetics and haemodynamic effects of norepinephrine
            # in hypotensive critically ill children,” British Journal of Clinical Pharmacology,
            # vol. 78, no. 4, pp. 886–897, 2014, doi: 10.1111/bcp.12412.
            self.emax_nore_map = 32
            self.c50_nore_map = 4.11
            self.gamma_nore_map = 1
            w_emax_nore_map = 0
            w_c50_nore_map = 0.09
            w_gamma_nore_map = 0
        self.k_effect = 0.0002  # (1/s)

        if random:
            # compute intercorrelated uncertainties
            eta_values_block1 = np.random.multivariate_normal(self.w_block1_mu, self.w_block1_cov, size=1)[0]
            eta_values_block2 = np.random.multivariate_normal(self.w_block2_mu, self.w_block2_cov, size=1)[0]

            # lognormal distribution
            self.tpr_base *= np.exp(eta_values_block1[0])
            self.sv_base *= np.exp(eta_values_block1[1])
            self.hr_base *= np.exp(eta_values_block1[2])
            self.c50_propo_tpr *= np.exp(np.random.normal(0, self.w_c50_propo_tpr))
            # normal distribution
            self.emax_remi_tpr += np.random.normal(0, self.w_emax_remi_tpr)
            self.sl_remi_hr += eta_values_block2[0]
            self.sl_remi_sv += eta_values_block2[1]

            self.emax_nore_map *= np.exp(np.random.normal(0, scale=w_emax_nore_map))
            self.c50_nore_map *= np.exp(np.random.normal(0, scale=w_c50_nore_map))
            self.gamma_nore_map *= np.exp(np.random.normal(0, scale=w_gamma_nore_map))

        self.x = np.array([
            self.tpr_base,
            self.sv_base,
            self.hr_base,
            self.sv_base * self.ltde_sv,
            self.hr_base * self.ltde_hr,
        ])

        self.x_effect = np.array([
            self.tpr_base,
            self.sv_base,
            self.hr_base,
            self.sv_base * self.ltde_sv,
            self.hr_base * self.ltde_hr,
        ])
        self.flag_nore_used = False
        self.flag_blood_loss = False

        self.abase_sv = self.sv_base * (1 + self.ltde_sv)
        self.abase_hr = self.hr_base * (1 + self.ltde_hr)
        self.base_map = self.tpr_base * self.abase_sv * self.abase_hr

        self.previous_cp_propo = 0
        self.previous_cp_remi = 0

        # intubation stimuli filter
        self.smooth_factor = 50
        self.dist_intub_tpr_tau = 1
        self.dist_intub_tpr_k = 1e-6
        self.dist_intub_sv_tau = 1
        self.dist_intub_sv_k = 1e-4
        self.dist_intub_hr_tau = 1
        self.dist_intub_hr_k = 1e-4

        # intubation stimuli filter
        # self.smooth_factor = 50
        self.dist_surg_tpr_tau = 1
        self.dist_surg_tpr_k = 1e-6
        self.dist_surg_sv_tau = 1
        self.dist_surg_sv_k = 1e-4
        self.dist_surg_hr_tau = 1
        self.dist_surg_hr_k = 1e-4

        # stimuli input
        self.intub_input = np.zeros(10000)
        self.intub_input[4*60//self.ts:6*60//self.ts] = 1  # step disturbance after 5 seconds
        self.surg_input = np.zeros(10000)
        self.surg_input[40*60//self.ts:] = 1  # step disturbance after 5 seconds
        self.time_id = 0

        self.disturbance_dynamic()

    def disturbance_dynamic(self):
        """Apply the dynamic on the input disturbance signal.
        """
        smooth_filter = TransferFunction(
            [1],
            [self.smooth_factor, 1]
        ).to_discrete(self.ts, method='bilinear')
        tf_dist_tpr = TransferFunction(
            [self.dist_intub_tpr_k],
            [self.dist_intub_tpr_tau, 1, 0]
        ).to_discrete(self.ts, method='bilinear')
        tf_dist_sv = TransferFunction(
            [self.dist_intub_sv_k],
            [self.dist_intub_sv_tau, 1, 0]
        ).to_discrete(self.ts, method='bilinear')
        tf_dist_hr = TransferFunction(
            [self.dist_intub_hr_k],
            [self.dist_intub_hr_tau, 1]
        ).to_discrete(self.ts, method='bilinear')

        input_smooth = lfilter(smooth_filter.num, smooth_filter.den, self.intub_input)
        self.dist_tpr = lfilter(tf_dist_tpr.num, tf_dist_tpr.den, input_smooth)
        self.dist_sv = lfilter(tf_dist_sv.num, tf_dist_sv.den, input_smooth)
        self.dist_hr = lfilter(tf_dist_hr.num, tf_dist_hr.den, input_smooth)

        tf_dist_tpr = TransferFunction(
            [self.dist_surg_tpr_k],
            [self.dist_surg_tpr_tau**2, 2*self.dist_surg_tpr_tau, 1]
        ).to_discrete(self.ts, method='bilinear')
        tf_dist_sv = TransferFunction(
            [self.dist_surg_sv_k],
            [self.dist_surg_sv_tau**2, 2*self.dist_surg_sv_tau, 1]
        ).to_discrete(self.ts, method='bilinear')
        tf_dist_hr = TransferFunction(
            [self.dist_surg_hr_k],
            [self.dist_surg_hr_tau**2, 2*self.dist_surg_hr_tau, 1]
        ).to_discrete(self.ts, method='bilinear')

        self.dist_tpr += lfilter(tf_dist_tpr.num, tf_dist_tpr.den, self.surg_input)
        self.dist_sv += lfilter(tf_dist_sv.num, tf_dist_sv.den, self.surg_input)
        self.dist_hr += lfilter(tf_dist_hr.num, tf_dist_hr.den, self.surg_input)

    def continuous_dynamic(
            self,
            x: np.ndarray,
            u: np.ndarray
    ) -> np.ndarray:
        """Define the continuous dynamic of the haemodynamic system.

        For implementation details see supplementary material nb 6 of the paper of Su and co-authors.

        Parameters
        ----------
        x : np.ndarray
            state array composed of tpr, sv, hr, ltde_sv, ltde_hr.
        u: np.ndarray
            u = [cp_propo, cp_remi, map_wanted, sv_wanted], plasma concentration of propofol (µg/ml) and remifentanil (ng/ml).

        Returns
        -------
        d x / dt : np.ndarray
            temporal derivative of the state array.
        """

        # compute drug effect
        cp_propo = u[0]
        cp_remi = u[1]
        map_wanted = u[2]
        sv_wanted = u[3]
        tpr_stim = u[4]
        sv_stim = u[5]
        hr_stim = u[6]

        eff_propo_tpr = (self.emax_propo_tpr + self.int_tpr * fsig(cp_remi, self.c50_remi_tpr, 1)) * \
            fsig(cp_propo, self.c50_propo_tpr, self.gamma_propo_tpr)
        eff_remi_sv = (self.sl_remi_sv + self.int_sv * fsig(cp_propo, self.c50_propo_sv, 1)) * cp_remi
        eff_remi_hr = (self.sl_remi_hr + self.int_hr * fsig(cp_propo, self.c50_int_hr, 1)) * cp_remi
        eff_propo_sv = self.emax_propo_sv * fsig(cp_propo, self.c50_propo_sv, 1)
        eff_remi_tpr = self.emax_remi_tpr * fsig(cp_remi, self.c50_remi_tpr, 1)

        # compute apparent values
        dtpr = x[0] + tpr_stim
        dsv = x[1] + x[3] + sv_stim
        dhr = x[2] + x[4] + hr_stim

        a_sv = dsv * (1 - self.hr_sv * np.log(dhr / self.abase_hr))
        a_map = a_sv * dhr * dtpr

        rmap = a_map / self.base_map

        sv = x[1]
        hr = x[2]
        tpr_dot = self.k_in_tpr * rmap**self.fb * (1 + eff_propo_tpr) - \
            self.k_out * x[0] * (1 - eff_remi_tpr)
        if map_wanted > 0:
            tpr_dot += (map_wanted - a_map) * self.k_effect
        sv_dot_star = self.k_in_sv * rmap**self.fb * (1 + eff_propo_sv) - self.k_out * sv * (1 - eff_remi_sv)
        if sv_wanted > 0:
            sv_dot_star += (sv_wanted - a_sv) * self.k_effect * 10000
        hr_dot_star = self.k_in_hr * rmap**self.fb - self.k_out * hr * (1 - eff_remi_hr)

        if u[0] > 0:  # apply the time dependant function only if anesthesia as started.
            ltde_sv_dot = -self.k_ltde * x[3]
            ltde_hr_dot = -self.k_ltde * x[4]
        else:
            ltde_sv_dot = 0
            ltde_hr_dot = 0

        return np.array([tpr_dot, sv_dot_star, hr_dot_star, ltde_sv_dot, ltde_hr_dot])

    def continuous_dynamic_sys(self, t, x, u):
        """ Same as continuous_dynamic but with time as first arguments (for scipy simulation)."""
        return self.continuous_dynamic(x, u)

    def output_function(self, x: np.ndarray, dist: np.ndarray = np.zeros(3)) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        x : np.ndarray
            state of the system

        Returns
        -------
        np.ndarray
            Total peripheral resistance (mmHg min/ mL), Stroke volume (ml),
            heart rate (beat / min), mean arterial pressure (mmHg),
            cardiac output (L/min)
        """
        tpr = x[0] + dist[0]
        sv = x[1] + x[3] + dist[1]
        hr = x[2] + x[4] + dist[2]
        sv = sv * (1 - self.hr_sv * np.log(hr / self.abase_hr))
        map = tpr * sv * hr
        co = hr * sv / 1000  # fro mL/min to L/min
        return np.array([tpr, sv, hr, map, co])

    def nore_map_effect(self, cp_nore: float):
        """Compute Norepinephrine effect on MAP.

        Parameters
        ----------
        c_nore : float
            Concentration of Norepinephrine (ng/mL)
        """

        return self.emax_nore_map * fsig(cp_nore, self.c50_nore_map, self.gamma_nore_map)

    def one_step(
            self,
            cp_propo: float = 0,
            cp_remi: float = 0,
            cp_nore: float = 0,
            v_ratio: float = 1,
    ) -> np.ndarray:
        """Compute one step time of the hemodynamic system.

        It use Runge Kutta 4 to compute the non-linear integration.

        Parameters
        ----------
        c_propo : float
            current plasma concentration of propofol (µg/ml), default is 0.
        cp_remi : float
            current plasma concentration of remifentanil (ng/ml), default is 0.
        cp_nore : float
            current plasma concentration of norepinephrine (ng/ml), default is 0.
        v_ratio : float
            blood volume as a fraction of init volume, 1 mean no loss, 0 mean 100% loss, default is 1.
        """
        # simulate disturbance
        dist_tpr = self.dist_tpr[self.time_id]
        dist_sv = self.dist_sv[self.time_id]
        dist_hr = self.dist_hr[self.time_id]
        dist = [dist_tpr, dist_sv, dist_hr]
        self.time_id += 1
        if self.time_id >= len(self.intub_input):
            self.time_id = len(self.intub_input) - 1

        c_propo_sim = (self.previous_cp_propo + cp_propo) / 2
        c_remi_sim = (self.previous_cp_remi + cp_remi) / 2
        # run computation for model without nore effect and without blood loss
        results = solve_ivp(
            self.continuous_dynamic_sys,
            t_span=np.array([0, self.ts]),
            t_eval=np.array([0, self.ts]),
            y0=self.x,
            args=([c_propo_sim, c_remi_sim, 0, 0]+dist,),
        )
        self.x = results.y[:, -1]

        if (self.flag_nore_used or cp_nore > 0) and not self.flag_blood_loss:
            if not self.flag_blood_loss:
                self.flag_nore_used = True
            if v_ratio != 1:
                print("Warning: norepinephrine effect is not computed with blood loss")
            map_no_nore = self.output_function(self.x, dist)[3]
            map_wanted = map_no_nore + self.nore_map_effect(cp_nore)
            # run computation for model with nore effect
            results_w_nore = solve_ivp(
                self.continuous_dynamic_sys,
                t_span=np.array([0, self.ts]),
                y0=self.x_effect,
                args=([c_propo_sim, c_remi_sim, map_wanted, 0]+dist,),
            )
            self.x_effect = results_w_nore.y[:, -1]
        elif (v_ratio < 1 or self.flag_blood_loss) and not self.flag_nore_used:
            if not self.flag_blood_loss:
                self.flag_blood_loss = True
            if cp_nore > 0:
                print("Warning: norepinephrine effect is not computed with blood loss")
            sv_no_blood_loss = self.output_function(self.x, dist)[1]
            sv_wanted = sv_no_blood_loss * v_ratio
            results_blood_loss = solve_ivp(
                self.continuous_dynamic_sys,
                t_span=np.array([0, self.ts]),
                y0=self.x_effect,
                args=([c_propo_sim, c_remi_sim, 0, sv_wanted]+dist,),
            )
            self.x_effect = results_blood_loss.y[:, -1]
        else:
            self.x_effect = self.x.copy()

        self.previous_cp_propo = cp_propo
        self.previous_cp_remi = cp_remi
        output = self.output_function(self.x_effect, dist)
        return output  # tpr, sv, hr, map, co

    def full_sim(self,
                 cp_propo: np.ndarray,
                 cp_remi: np.ndarray,
                 cp_nore: np.ndarray,
                 x0: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """ Simulate hemodynamic model with a given input.

        Parameters
        ----------
        c_propo : np.ndarray
            list of plasma concentration of propofol (µg/ml).
        cp_remi : np.ndarray
            list of plasma concentration of remifentanil (ng/ml).
        cp_nore : np.ndarray
            list of plasma concentration of norepinephrine (ng/ml). 
        x0 : np.ndarray, optional
            Initial state. The default is None.

        Returns
        -------
        np.ndarray
            List of the output value during the simulation.
        """
        if len(cp_remi) != len(cp_propo) or len(cp_remi) != len(cp_nore):
            raise ValueError("inputs must have the same lenght")
        if x0 is not None:
            self.x = x0
            self.x_no_nore = x0

        y_output = np.zeros((len(cp_propo), 5))
        for index in range(len(cp_propo)):
            y_output[index, :] = self.one_step(cp_propo[index], cp_remi[index], cp_nore[index])

        return y_output

    def state_at_equilibrium(
            self,
            cp_propo_eq: float = 0,
            cp_remi_eq: float = 0,
            cp_nore_eq: float = 0,
            x0: np.ndarray = None,
    ) -> np.ndarray:
        """Solve the problem f(x,u)=0 for the continuous dynamique with a given u.

        Parameters
        ----------
        c_propo : float
            plasma concentration of propofol at equilibrium (µg/ml).
        cp_remi : float
            plasma concentration of remifentanil  at equilibrium (ng/ml).
        cp_nore : float
            plasma concentration of norepinephrine  at equilibrium (ng/ml).
        x0 : np.ndarray, optional
            Initial state. The default is None.

        Returns
        -------
        np.ndarray
            List of the output value at equilibrium.
        """

        if x0 is None:
            x0 = self.x

        # solve equilibrium without nore
        x = cas.MX.sym('x', 5)
        dx = cas.vertcat(*self.continuous_dynamic(x, [cp_propo_eq, cp_remi_eq, 0, 0]))
        F_root = cas.rootfinder('F_root', 'newton', {'x': x, 'g': dx})
        sol = F_root(x0=x0)
        x_no_nore = sol['x'].full().flatten()
        self.x_eq = x_no_nore

        # if nore is used, solve equilibrium with nore
        if cp_nore_eq > 0:
            output_no_nore = self.output_function(x_no_nore)
            map_eq = output_no_nore[3] + self.nore_map_effect(cp_nore_eq)
            # solve equilibrium with nore
            x = cas.MX.sym('x', 5)
            dx = cas.vertcat(*self.continuous_dynamic(x, [cp_propo_eq, cp_remi_eq, map_eq, 0]))
            # map_nore = self.output_function(x)[3]
            # dx[0] = (map_nore - map_eq)**2
            F_root = cas.rootfinder('F_root', 'newton', {'x': x, 'g': dx})
            sol = F_root(x0=x0)

            x_eq_out = sol['x'].full().flatten()
        else:
            x_eq_out = x_no_nore

        output = self.output_function(x_eq_out)
        self.x_eq_w_nore = x_eq_out

        return output

    def initialized_at_given_concentration(
            self,
            cp_propo_eq: float = 0,
            cp_remi_eq: float = 0,
            cp_nore_eq: float = 0
    ) -> None:
        """Initialize the haemodynamic model at a given concentration.

        Parameters
        ----------
        c_propo : float
            plasma concentration of propofol at equilibrium (µg/ml).
        cp_remi : float
            plasma concentration of remifentanil  at equilibrium (ng/ml).
        cp_nore : float
            plasma concentration of norepinephrine  at equilibrium (ng/ml).

        Returns
        -------
        None
            The haemodynamic model is initialized at the given concentration.
        """

        # compute the state at equilibrium
        _ = self.state_at_equilibrium(cp_propo_eq, cp_remi_eq, cp_nore_eq)
        # initialize the haemodynamic model
        self.x_effect = self.x_eq_w_nore
        self.x = self.x_eq
        self.previous_cp_propo = cp_propo_eq
        self.previous_cp_remi = cp_remi_eq
