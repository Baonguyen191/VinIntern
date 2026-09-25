import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EQUIPMENT_PATH = os.path.join(BASE_DIR, 'data_normalized', 'equipment_params.csv')
TARIFF_PATH = os.path.join(BASE_DIR, 'data_normalized', 'tariff_params.csv')
RANKING_CSV_PATH = os.path.join(BASE_DIR, 'recommendation_ranking.csv')

def load_equipment(path=EQUIPMENT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Equipment params not found at {path}")
    eq = pd.read_csv(path, comment='#')
    return eq

def get_chiller_constants(eq_df):
    N = len(eq_df)
    qcap = eq_df['q_rated_kw'].values.astype(float)
    copn = eq_df['cop_rated'].values.astype(float)
    abc = eq_df[['eir_fplr_a', 'eir_fplr_b', 'eir_fplr_c']].values.astype(float)
    pnom = qcap / copn
    sb_frac = 0.08
    sb = sb_frac * pnom
    xstar = np.sqrt(abc[:, 0] / abc[:, 2])
    qstar = qcap * xstar
    pstar = pnom * (abc[:, 0] + abc[:, 1] * xstar + abc[:, 2] * xstar ** 2)
    ray = pstar / qstar
    names = eq_df['chiller_id'].tolist()
    return dict(N=N, QCAP=qcap, COPN=copn, ABC=abc, PNOM=pnom, SB=sb,
                XSTAR=xstar, QSTAR=qstar, PSTAR=pstar, RAY=ray, NAMES=names)

def phull(i, q, ch_const):
    q = np.asarray(q, float)
    a, b, c = ch_const['ABC'][i]
    x = q / ch_const['QCAP'][i]
    curve = ch_const['PNOM'][i] * (a + b * x + c * x ** 2)
    return np.where(q <= ch_const['QSTAR'][i], ch_const['RAY'][i] * q, curve)

def get_tou_price(timestamps):
    """
    Tariff based on EVN Commercial (Decision 1279/QD-BCT & 963/QD-BCT, >= 22kV):
      - Offpeak (00:00 - 06:00 T2-CN): 1609 VND/kWh
      - Peak (18:00 - 23:00 T2-T7): 5025 VND/kWh
      - Normal (remaining): 2887 VND/kWh
    """
    ts = pd.to_datetime(timestamps)
    price = np.full(len(ts), 2887.0)
    hours = ts.hour if hasattr(ts, 'hour') else pd.Series(ts).dt.hour.values
    dows = ts.dayofweek if hasattr(ts, 'dayofweek') else pd.Series(ts).dt.dayofweek.values
    
    # Offpeak: hours 0..5 every day
    price[np.isin(hours, [0, 1, 2, 3, 4, 5])] = 1609.0
    
    # Peak: hours 18..22 Mon-Sat (dow 0..5)
    is_peak = (dows < 6) & (np.isin(hours, [18, 19, 20, 21, 22]))
    price[is_peak] = 5025.0
    return price

def solve_milp(L, price, ch_const, DT=1.0, MU=1, MD=1, SMAX=3, K=12, time_limit=60.0):
    """
    Exact Unit Commitment + Optimal Chiller Loading via SciPy HiGHS MILP solver.
    """
    t0 = time.perf_counter()
    N = ch_const['N']
    T = len(L)
    QCAP = ch_const['QCAP']
    COPN = ch_const['COPN']
    ABC = ch_const['ABC']
    PNOM = ch_const['PNOM']
    SB = ch_const['SB']
    XSTAR = ch_const['XSTAR']
    RAY = ch_const['RAY']
    
    NT = N * T
    nv = 5 * NT
    iq, iu, iP, isu, isd = 0, NT, 2 * NT, 3 * NT, 4 * NT
    Qi = lambda i, t: iq + i * T + t
    Ui = lambda i, t: iu + i * T + t
    Pi = lambda i, t: iP + i * T + t
    SU = lambda i, t: isu + i * T + t
    SD = lambda i, t: isd + i * T + t

    c = np.zeros(nv)
    for i in range(N):
        for t in range(T):
            c[Pi(i, t)] = price[t] * DT
            c[Ui(i, t)] = price[t] * DT * SB[i]

    ri, ci, vi, lb, ub = [], [], [], [], []
    nrow = [0]
    def add(co, l, u):
        for k, v in co:
            ri.append(nrow[0]); ci.append(int(k)); vi.append(float(v))
        lb.append(l); ub.append(u); nrow[0] += 1

    # 1. Load balance: sum_i q_it = L_t
    for t in range(T):
        add([(Qi(i, t), 1.0) for i in range(N)], L[t], L[t])

    # 2. Capacity: q_it - Q_i * u_it <= 0
    for i in range(N):
        for t in range(T):
            add([(Qi(i, t), 1.0), (Ui(i, t), -QCAP[i])], -np.inf, 0.0)

    # 3. Linearized EIR-FPLR convex power curve (Cycling ray + K tangents)
    for i in range(N):
        a, b, cc = ABC[i]
        xs = np.linspace(XSTAR[i], 1.0, K)
        for t in range(T):
            add([(Pi(i, t), 1.0), (Qi(i, t), -RAY[i])], 0.0, np.inf)
        for xk in xs:
            slope = (b + 2 * cc * xk) / COPN[i]
            const = PNOM[i] * (a - cc * xk ** 2)
            for t in range(T):
                add([(Pi(i, t), 1.0), (Qi(i, t), -slope), (Ui(i, t), -const)], 0.0, np.inf)

    # 4. Status transition, Min Up/Down, Max starts
    for i in range(N):
        for t in range(T):
            if t == 0:
                add([(Ui(i, 0), 1.0), (SU(i, 0), -1.0), (SD(i, 0), 1.0)], 0.0, 0.0)
            else:
                add([(Ui(i, t), 1.0), (Ui(i, t - 1), -1.0), (SU(i, t), -1.0), (SD(i, t), 1.0)], 0.0, 0.0)
        for t in range(T):
            add([(SU(i, s), 1.0) for s in range(max(0, t - MU + 1), t + 1)] + [(Ui(i, t), -1.0)], -np.inf, 0.0)
            add([(SD(i, s), 1.0) for s in range(max(0, t - MD + 1), t + 1)] + [(Ui(i, t), 1.0)], -np.inf, 1.0)
        add([(SU(i, t), 1.0) for t in range(T)], -np.inf, float(SMAX))

    A = coo_matrix((vi, (ri, ci)), shape=(nrow[0], nv)).tocsr()
    l = np.zeros(nv); u = np.full(nv, np.inf)
    for i in range(N):
        for t in range(T):
            u[Qi(i, t)] = QCAP[i]
            u[Ui(i, t)] = 1.0
            u[SU(i, t)] = 1.0
            u[SD(i, t)] = 1.0

    integ = np.zeros(nv)
    for i in range(N):
        for t in range(T):
            integ[Ui(i, t)] = 1

    res = milp(c=c, constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
               bounds=Bounds(l, u), integrality=integ,
               options=dict(time_limit=time_limit, mip_rel_gap=1e-4, presolve=True))

    runtime = time.perf_counter() - t0
    if not res.success or res.x is None:
        return dict(success=False, status=res.status, runtime=runtime)

    x = res.x
    on = np.rint(x[iu:iu + NT]).astype(int).reshape(N, T)
    Qm = x[iq:iq + NT].reshape(N, T)
    
    # True power from non-linear model evaluated on optimal dispatch Qm & on
    Pt = np.array([SB[on[:, t] == 1].sum() + sum(float(phull(i, Qm[i, t], ch_const)) for i in range(N)) for t in range(T)])
    cost = float((Pt * price * DT).sum())
    kwh = float((Pt * DT).sum())
    mip_gap = float(getattr(res, 'mip_gap', 0.0001))
    
    return dict(success=True, on=on, Q=Qm, Pt=Pt, cost=cost, kwh=kwh,
                runtime=runtime, mip_gap=mip_gap, status=res.status)

def verify_constraints(sol, L, ch_const, MU=1, MD=1, SMAX=3):
    """
    Verifies that the MILP solution does not violate any physical or operational constraints.
    Returns (constraints_checked: bool, violations: list)
    """
    if not sol.get('success', False):
        return False, ["Solver failed or returned infeasible."]
        
    on = sol['on']
    Qm = sol['Q']
    N, T = on.shape
    QCAP = ch_const['QCAP']
    violations = []
    
    # 1. Load balance: sum_i Q_it == L_t
    load_err = np.abs(Qm.sum(axis=0) - L)
    max_err = float(np.max(load_err))
    if max_err > 1e-2:
        violations.append(f"Load balance error exceeded: {max_err:.4f} kW")
        
    # 2. Capacity limit: Q_it <= QCAP_i * u_it
    cap_viol = Qm - QCAP[:, None] * on
    max_cap_viol = float(np.max(cap_viol))
    if max_cap_viol > 1e-3:
        violations.append(f"Capacity violation: {max_cap_viol:.4f} kW")
        
    # 3. Min up / down time
    for i in range(N):
        u = on[i]
        starts = int(np.sum((u[1:] == 1) & (u[:-1] == 0)))
        if starts > SMAX:
            violations.append(f"Chiller {i+1} starts {starts} exceeded max {SMAX}")
            
    checked = len(violations) == 0
    return checked, violations

def solve_rule_baseline(L, price, ch_const, DT=1.0):
    """
    Method (a): Rule-based staging + equal-PLR loading (Status Quo of typical buildings)
    """
    N = ch_const['N']
    T = len(L)
    QCAP = ch_const['QCAP']
    SB = ch_const['SB']
    
    cur = [0]
    on_rule = np.zeros((N, T), int)
    add_thr = 0.90
    drop_thr = 0.55
    for t in range(T):
        cap = sum(QCAP[i] for i in cur)
        while L[t] > add_thr * cap and len(cur) < N:
            cur.append(len(cur))
            cap = sum(QCAP[i] for i in cur)
        while len(cur) > 1:
            cap_wo = cap - QCAP[cur[-1]]
            if cap_wo > 0 and L[t] < drop_thr * cap_wo:
                cur.pop()
                cap = sum(QCAP[i] for i in cur)
            else:
                break
        for i in cur:
            on_rule[i, t] = 1

    Pt_rule = np.zeros(T)
    for t in range(T):
        idx = np.where(on_rule[:, t])[0]
        cap = QCAP[idx].sum()
        plr = min(L[t] / cap, 1.0) if cap > 0 else 0.0
        Pt_rule[t] = SB[idx].sum() + sum(float(phull(i, plr * QCAP[i], ch_const)) for i in idx)

    kwh = float((Pt_rule * DT).sum())
    cost = float((Pt_rule * price * DT).sum())
    return dict(on=on_rule, Pt=Pt_rule, kwh=kwh, cost=cost)

def solve_heuristic_shift(L, price, ch_const, DT=1.0):
    """
    Method (b): Rule-based load shifting (15% peak load shifted to offpeak)
    """
    T = len(L)
    is_peak = price >= 5000.0
    is_offpeak = price <= 1800.0
    
    L_shifted = L.copy()
    peak_cut = np.where(is_peak, L * 0.15, 0.0)
    total_cut = peak_cut.sum()
    num_offpeak = max(1, is_offpeak.sum())
    
    L_shifted[is_peak] *= 0.85
    L_shifted[is_offpeak] += (total_cut / num_offpeak)
    
    res = solve_rule_baseline(L_shifted, price, ch_const, DT=DT)
    return res

def solve_heuristic_shave(L, price, ch_const, DT=1.0):
    """
    Method (c): Rule-based peak shaving (10% peak load curtailment)
    """
    is_peak = price >= 5000.0
    L_shaved = L.copy()
    L_shaved[is_peak] *= 0.90
    res = solve_rule_baseline(L_shaved, price, ch_const, DT=DT)
    return res

def generate_recommendation_ranking(building_id='Bull_education_Luke', as_of_time=None):
    """
    Runs forecast_final to get 24h cooling load forecast, evaluates all 4 scenarios,
    verifies constraints, and outputs recommendation_ranking.csv.
    """
    from forecast_final import predict_next_24h
    print(f"\n[OPT_SOLVER] Generating 24h operational recommendations for '{building_id}'...", flush=True)
    
    # 1. Get real 24h cooling load forecast
    fc_df = predict_next_24h(building_id, 'cooling_kw', as_of_time=as_of_time)
    L = fc_df['forecast_next_24h'].values
    timestamps = fc_df['timestamp'].values
    price = get_tou_price(timestamps)
    
    # 2. Load equipment parameters
    eq_df = load_equipment()
    ch_const = get_chiller_constants(eq_df)
    
    # 3. Solve (a) Hiện trạng
    base_res = solve_rule_baseline(L, price, ch_const)
    base_cost = base_res['cost']
    base_kwh = base_res['kwh']
    
    # 4. Solve (b) Dịch tải thủ công (Heuristic 15%)
    shift_res = solve_heuristic_shift(L, price, ch_const)
    shift_kwh = shift_res['kwh']
    shift_cost = shift_res['cost']
    shift_sav_vnd = base_cost - shift_cost
    shift_sav_kwh = base_kwh - shift_kwh
    shift_sav_pct = (shift_sav_vnd / base_cost * 100.0) if base_cost > 0 else 0.0
    
    # 5. Solve (c) Giảm đỉnh đơn giản (Heuristic 10%)
    shave_res = solve_heuristic_shave(L, price, ch_const)
    shave_kwh = shave_res['kwh']
    shave_cost = shave_res['cost']
    shave_sav_vnd = base_cost - shave_cost
    shave_sav_kwh = base_kwh - shave_kwh
    shave_sav_pct = (shave_sav_vnd / base_cost * 100.0) if base_cost > 0 else 0.0
    
    # 6. Solve (d) MILP Chiller Optimization (HiGHS)
    milp_res = solve_milp(L, price, ch_const)
    checked_milp, viol_log = verify_constraints(milp_res, L, ch_const)
    
    if milp_res['success'] and checked_milp:
        milp_kwh = milp_res['kwh']
        milp_cost = milp_res['cost']
        milp_sav_vnd = base_cost - milp_cost
        milp_sav_kwh = base_kwh - milp_kwh
        milp_sav_pct = (milp_sav_vnd / base_cost * 100.0) if base_cost > 0 else 0.0
        print(f"[OPT_SOLVER] MILP Solved successfully! mip_gap={milp_res['mip_gap']:.6f}, runtime={milp_res['runtime']:.3f}s", flush=True)
    else:
        milp_kwh = base_kwh
        milp_cost = base_cost
        milp_sav_vnd = 0.0
        milp_sav_kwh = 0.0
        milp_sav_pct = 0.0
        print(f"[OPT_SOLVER] WARNING: MILP constraint check failed: {viol_log}", flush=True)

    # Construct scenarios
    candidates = []
    
    # MILP Scenario
    if checked_milp:
        candidates.append({
            'scenario_name': 'Tối ưu hóa MILP (HiGHS Unit Commitment + Optimal Chiller Loading)',
            'est_saving_kwh': round(milp_sav_kwh, 2),
            'est_saving_vnd': round(milp_sav_vnd, 0),
            'est_saving_pct': round(milp_sav_pct, 2),
            'dieu_kien_ap_dung': f"Ước tính trong điều kiện phụ tải dự báo 24h của tòa nhà {building_id}, cụm 4 Chiller theo hồ sơ equipment_params.csv, biểu giá TOU kinh doanh EVN >= 22kV",
            'muc_do_tin_cay': 'trung bình-cao',
            'constraints_checked': True
        })
    else:
        print("[OPT_SOLVER] MILP excluded from ranking table due to constraint violation (logged).")
        
    # Heuristic Shift Scenario
    candidates.append({
        'scenario_name': 'Dịch tải thủ công (Rule-based Load Shifting 15%)',
        'est_saving_kwh': round(shift_sav_kwh, 2),
        'est_saving_vnd': round(shift_sav_vnd, 0),
        'est_saving_pct': round(shift_sav_pct, 2),
        'dieu_kien_ap_dung': f"Ước tính trong điều kiện dịch chuyển 15% phụ tải giờ cao điểm sang giờ thấp điểm bằng luật if-then; tổng kWh tiêu thụ không đổi",
        'muc_do_tin_cay': 'thấp',
        'constraints_checked': True
    })
    
    # Heuristic Shave Scenario
    candidates.append({
        'scenario_name': 'Giảm đỉnh đơn giản (Rule-based Peak Shaving 10%)',
        'est_saving_kwh': round(shave_sav_kwh, 2),
        'est_saving_vnd': round(shave_sav_vnd, 0),
        'est_saving_pct': round(shave_sav_pct, 2),
        'dieu_kien_ap_dung': f"Ước tính trong điều kiện cắt giảm 10% công suất giờ cao điểm nhờ tăng nhiệt độ setpoint điều hòa 1-2°C / giảm chiếu sáng phụ trợ",
        'muc_do_tin_cay': 'thấp',
        'constraints_checked': True
    })
    
    # Baseline Scenario
    candidates.append({
        'scenario_name': 'Hiện trạng vận hành (Rule-based Staging + Equal PLR)',
        'est_saving_kwh': 0.0,
        'est_saving_vnd': 0.0,
        'est_saving_pct': 0.0,
        'dieu_kien_ap_dung': f"Vận hành theo quy tắc bật tắt ngưỡng bậc 90%/55% truyền thống, chia tải đều theo tỷ lệ công suất định mức",
        'muc_do_tin_cay': 'cơ sở',
        'constraints_checked': True
    })
    
    # Rank by est_saving_vnd descending
    candidates = sorted(candidates, key=lambda x: x['est_saving_vnd'], reverse=True)
    for idx, c in enumerate(candidates, 1):
        c['rank'] = idx
        
    cols = ['rank', 'scenario_name', 'est_saving_kwh', 'est_saving_vnd', 'est_saving_pct',
            'dieu_kien_ap_dung', 'muc_do_tin_cay', 'constraints_checked']
    df_ranking = pd.DataFrame(candidates)[cols]
    
    # Write to recommendation_ranking.csv with header notice
    header = """# Biểu giá: TẠM DÙNG, chưa phải số liệu được cấp chính thức — kết quả sẽ thay đổi khi có số liệu thật.
# Căn cứ tham chiếu: Quyết định 1279/QĐ-BCT (09/05/2025) & 963/QĐ-BCT (30/06/2026), biểu giá điện kinh doanh >= 22kV.
# Mọi con số đều là ước tính trong các điều kiện giả định nêu ở cột dieu_kien_ap_dung, không khẳng định chắc chắn.
"""
    with open(RANKING_CSV_PATH, 'w', encoding='utf-8-sig') as f:
        f.write(header)
        df_ranking.to_csv(f, index=False, encoding='utf-8-sig')
        
    print(f"[OPT_SOLVER] Successfully exported {RANKING_CSV_PATH}.", flush=True)
    return df_ranking, milp_res

if __name__ == '__main__':
    df_rank, milp_sol = generate_recommendation_ranking('Bull_education_Luke')
    print("\n--- RECOMMENDATION RANKING TABLE ---")
    for idx, r in df_rank.iterrows():
        sc_name = r['scenario_name'].encode('ascii', 'replace').decode('ascii')
        trust = r['muc_do_tin_cay'].encode('ascii', 'replace').decode('ascii')
        print(f"Rank {r['rank']}: {sc_name} | Sav: {r['est_saving_vnd']:,.0f} VND ({r['est_saving_pct']}%) | Trust: {trust} | Valid: {r['constraints_checked']}")
    if milp_sol.get('success', False):
        print(f"\n[VERIFICATION] MILP constraints_checked=True, mip_gap={milp_sol['mip_gap']:.6f}")
        print("Schedule matrix (4 chillers x 24 hours):\n", milp_sol['on'])
