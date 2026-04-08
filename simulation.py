import mph
import numpy as np
import pandas as pd
import time
import csv
import os
import multiprocessing as mp
import traceback

# 定义需要导入的函数和类
from txt2csv import txt_to_csv

def get_T2(model,  ts_txt_path=None, ts_csv_path=None, outlet_coords=(0.0, -200.0, 10000.0)):
    """
    弃用原先只读取表格的实现，改为：
    - 导出整个管道的 TXT（使用 model.exports()[0]），
    - 将 TXT 转换为标准 CSV（使用 txt_to_csv），
    - 在 CSV 中定位出水口 (默认 X=0,Y=-200,Z=10000)，先严格匹配，找不到再做最近邻匹配，
    - 返回 (times, temperatures_in_Celsius)；times 为 0..N-1。

    可选参数:
    - ts_txt_path: 导出 TXT 文件保存路径（若 None 则使用临时文件名）
    - ts_csv_path: 转换后 CSV 文件保存路径（若 None 则与 TXT 同名 .csv）
    - outlet_coords: (X,Y,Z) 出水口坐标用于定位行
    """
    try:
        # 1) 准备文件路径
        if ts_txt_path is None:
            # 若没有提供，则在当前工作目录下生成临时文件名
            ts_txt_path = os.path.join(os.getcwd(), f"pipeline_export_{int(time.time())}.txt")
        if ts_csv_path is None:
            ts_csv_path = os.path.splitext(ts_txt_path)[0] + '.csv'

        # 2) 获取导出项并写出 TXT（忽略 model.export 返回值为 NaN 的情况）
        exports = list(model.exports()) if hasattr(model, 'exports') else []
        if not exports:
            raise RuntimeError("模型中找不到导出项 (model.exports() 为空)")
        export_tag = exports[0]

        try:
            # 若导出失败会抛异常
            model.export(export_tag, ts_txt_path)
        except Exception as e:
            # 记录并继续尝试 — 有时 COMSOL 会返回 NaN 但仍生成文件
            # 我们允许后续通过存在的文件继续
            # 这里将错误包装成警告信息
            # print(f"导出警告: {e}")
            pass

        # 3) 等待导出文件出现（短暂轮询）
        wait_start = time.time()
        while not os.path.exists(ts_txt_path) and (time.time() - wait_start) < 10:
            time.sleep(0.2)
        if not os.path.exists(ts_txt_path):
            raise RuntimeError(f"导出文件未生成: {ts_txt_path}")

        # 4) 转换 TXT -> CSV
        try:
            txt_to_csv(ts_txt_path, output_path=ts_csv_path)
        except Exception as e:
            raise RuntimeError(f"TXT -> CSV 转换失败: {e}")

        # 5) 读取 CSV 并定位出水口行
        df = pd.read_csv(ts_csv_path)
        # 尝试直接匹配标准列名
        x_col = None
        for candidate in ['X', 'x', 'X_', 'X1']:
            if candidate in df.columns:
                x_col = candidate
                break
        # 更通用地查找坐标列名
        if x_col is None:
            possible = [c for c in df.columns if c.strip().lower() == 'x']
            x_col = possible[0] if possible else None
        y_col = None
        for candidate in ['Y', 'y', 'Y_', 'Y1']:
            if candidate in df.columns:
                y_col = candidate
                break
        if y_col is None:
            possible = [c for c in df.columns if c.strip().lower() == 'y']
            y_col = possible[0] if possible else None
        z_col = None
        for candidate in ['Z', 'z', 'Z_', 'Z1']:
            if candidate in df.columns:
                z_col = candidate
                break
        if z_col is None:
            possible = [c for c in df.columns if c.strip().lower() == 'z']
            z_col = possible[0] if possible else None

        if x_col is None or y_col is None or z_col is None:
            raise RuntimeError(f"无法在导出的 CSV 中找到坐标列 (查找 X/Y/Z) — 列: {df.columns.tolist()}")

        # 时间列：寻找以 'T_K_at_t_' 开头的列（txt2csv 的清洗规则会产生这样的列名）
        time_cols = [c for c in df.columns if c.startswith('T_K_at_t_')]
        if not time_cols:
            # 退路：任何以 'T' 开头且包含数字的列
            import re
            time_cols = [c for c in df.columns if c.startswith('T') and re.search(r'\d', c)]
        if not time_cols:
            raise RuntimeError("在导出的 CSV 中找不到时间序列列 (例如 'T_K_at_t_0')")

        # 先尝试严格匹配坐标
        X_val, Y_val, Z_val = outlet_coords
        coords_mask = np.isclose(df[x_col].astype(float).values, X_val) & np.isclose(df[y_col].astype(float).values, Y_val) & np.isclose(df[z_col].astype(float).values, Z_val)
        if coords_mask.any():
            row = df.loc[coords_mask].iloc[0]
        else:
            # 最近邻匹配
            dx = df[x_col].astype(float).values - X_val
            dy = df[y_col].astype(float).values - Y_val
            dz = df[z_col].astype(float).values - Z_val
            d2 = dx * dx + dy * dy + dz * dz
            idx = int(np.argmin(d2))
            row = df.iloc[idx]

        # 提取温度序列（开尔文）并转换为摄氏度
        temps_k = np.array([float(row[c]) for c in time_cols])
        temps_c = temps_k - 273.15

        # 解析时间列名（例如: T_K_at_t_0, T_K_at_t_0_01, T_K_at_t_1, ...）
        import re
        times_list = []
        for c in time_cols:
            m = re.match(r'T_K_at_t_(.+)', c)
            if m:
                s = m.group(1).replace('_', '.')
                try:
                    t_val = float(s)
                except Exception:
                    nums = re.findall(r'[-+]?\d*\.?\d+', s)
                    t_val = float(nums[-1]) if nums else np.nan
            else:
                nums = re.findall(r'[-+]?\d*\.?\d+', c)
                t_val = float(nums[-1]) if nums else np.nan
            times_list.append(t_val)

        times = np.array(times_list, dtype=float)
        # 若解析失败（出现 NaN），回退为索引 0..N-1
        if np.isnan(times).any():
            times = np.arange(len(time_cols), dtype=float)

        return times, temps_c

    except Exception as e:
        raise RuntimeError(f"get_T2 (pipeline export) 失败: {e}")

# ================= 配置区域 =================
N_SAMPLES = 200
CSV_FILE = "Data_10Params.csv" 
TIME_SERIES_DIR = "time_series_data"  # 存储时间序列数据的目录
TIMEOUT_SECONDS = 120  # 2分钟超时时间
# 在 Windows 上推荐使用子进程运行单次求解，以便在超时时能可靠终止进程。
USE_SUBPROCESS_SOLVE = True
def _solve_worker(model_path, params, ts_path, csv_path, q):
    """在子进程中加载模型、设置参数、求解并保存时间序列结果。"""
    try:
        client = mph.start()
        model = client.load(model_path)
        # 设置参数
        for k, v in params.items():
            model.parameter(k, v)
        model.reset()
        model.solve('研究 1')
        # 导出整个管道的数据到 TXT -> 转换为 CSV，再从 CSV 中读取出水口时间序列
        txt_path = ts_path.replace('.npz', '.txt')
        try:
            times, temps = get_T2(model, ts_txt_path=txt_path, ts_csv_path=csv_path)
        except Exception as e:
            raise

        tout_val = float(temps[-1])
        liquid_raw = model.evaluate('aveop1(ht2.alpha12)', inner='last')
        if hasattr(liquid_raw, 'item'):
            liquid_val = float(liquid_raw.item())
        elif isinstance(liquid_raw, (list, np.ndarray)):
            liquid_val = float(np.array(liquid_raw).flat[0])
        else:
            liquid_val = float(liquid_raw)

        # 保存时间序列（npz），并额外保存出口点的 Time/Temperature CSV 以便兼容旧逻辑
        np.savez(ts_path, times=times, temperatures=temps, parameters=params, results={
            'final_temp': tout_val,
            'liquid_fraction': liquid_val
        })
        outlet_csv = ts_path.replace('.npz', '.outlet.csv')
        pd.DataFrame({'Time': times, 'Temperature': temps}).to_csv(outlet_csv, index=False)

        q.put({'status': 'success', 'tout': tout_val, 'liquid': liquid_val})
        try:
            client.stop()
        except:
            pass
    except Exception:
        q.put({'status': 'error', 'error': traceback.format_exc()})


def solve_with_timeout(model_path, params, ts_path, csv_path, timeout_seconds, status_interval=30):
    """在子进程中运行求解，并在超时后终止进程。
    在等待期间每隔 `status_interval` 秒打印一次已用时间（例如 [30s][60s]）。
    返回包含状态和数据的字典。"""
    q = mp.Queue()
    p = mp.Process(target=_solve_worker, args=(model_path, params, ts_path, csv_path, q))
    p.start()

    start = time.time()
    elapsed = 0
    next_status = status_interval

    try:
        # 循环等待并每隔 status_interval 打印一次进度
        while p.is_alive() and elapsed < timeout_seconds:
            time.sleep(1)
            elapsed = time.time() - start
            if elapsed >= next_status:
                print(f"[{int(elapsed)}s]", end="", flush=True)
                next_status += status_interval

        # 结束等待后根据进程状态做处理
        if p.is_alive():
            p.terminate()
            p.join()
            print()  # 换行，结束进度提示行
            return {'status': 'timeout', 'elapsed': elapsed}
        else:
            print()  # 换行，结束进度提示行
            if not q.empty():
                res = q.get()
                res.setdefault('elapsed', elapsed)
                return res
            else:
                return {'status': 'error', 'error': 'No response from worker process', 'elapsed': elapsed}
    except Exception:
        # 出现异常时确保子进程被终止
        try:
            if p.is_alive():
                p.terminate()
                p.join()
        except:
            pass
        print()
        return {'status': 'error', 'error': traceback.format_exc(), 'elapsed': elapsed}

def run_simulation_batch():
    """主运行函数"""
    print("正在连接 COMSOL...")
    client = mph.start()
    print("正在加载模型...")
    model = client.load("DeepOnet.mph")
    print("✅ 模型加载完毕")
    
    print(f"计划生成 {N_SAMPLES} 组数据...")
    print(f"超时设置: {TIMEOUT_SECONDS}秒 ({TIMEOUT_SECONDS/60:.1f}分钟)")

    # --- 创建时间序列数据存储目录 ---
    if not os.path.exists(TIME_SERIES_DIR):
        os.makedirs(TIME_SERIES_DIR)
        print(f"✅ 已创建时间序列数据目录: {TIME_SERIES_DIR}")
    else:
        print(f"⚠️ 时间序列数据目录已存在: {TIME_SERIES_DIR}")

    # --- 生成 10 个材料参数的随机列表 ---
    # 使用当前时间作为随机种子，使每次运行结果不同
    seed = int(time.time())
    np.random.seed(seed)
    print(f"🔀 随机种子: {seed} （基于当前时间，保证每次运行不同）")

    # 1-10. 条件采样（保持物理约束：rho_s > rho_l，cl_l > cp_s，k_s >= k_l）
    # 说明：先采低序变量（rho_l, k_l），再采高序变量（rho_s, k_s）为其添加正的 delta，向量化实现。
    # 最小差值用于保证严格不等式（同时防止边界浮点问题）。
    min_delta_rho = 0.1   # kg/m^3
    min_delta_cp = 0.1    # J/(kg*K)
    min_delta_k = 1e-4    # W/(m*K)

    # 1) rho_l -> rho_s (保证 rho_s > rho_l 且 rho_s ∈ [800, 950])
    list_rho_l = np.random.uniform(750.0, 850.0, N_SAMPLES)
    min_rho_s = np.maximum(list_rho_l + min_delta_rho, 800.0)
    list_rho_s = np.random.uniform(min_rho_s, 950.0)

    # 2) cp_s -> cl_l (保证 cl_l > cp_s 且 cl_l ∈ [2000, 2600])
    list_cp_s = np.random.uniform(1800.0, 2400.0, N_SAMPLES)
    min_cl_l = np.maximum(list_cp_s + min_delta_cp, 2000.0)
    list_cl_l = np.random.uniform(min_cl_l, 2600.0)

    # 3) k_l -> k_s (保证 k_s >= k_l 且 k_s ∈ [0.200, 0.250])
    list_k_l   = np.random.uniform(0.140, 0.20, N_SAMPLES)
    min_k_s = np.maximum(list_k_l + min_delta_k, 0.200)
    list_k_s   = np.random.uniform(min_k_s, 0.250)

    # 其余参数保持原采样
    list_Lf    = np.random.uniform(150.0, 250.0, N_SAMPLES)
    list_wt    = np.random.uniform(0.10, 0.25, N_SAMPLES)
    list_Tm    = np.random.uniform(20.0, 25.0, N_SAMPLES)
    list_dT    = np.random.uniform(2.0, 7.0, N_SAMPLES)

    # 验证并修正极少数可能的违例（理论上不会发生，但以防浮点或边界问题）
    mask_rho = list_rho_s <= list_rho_l
    mask_cp = list_cl_l <= list_cp_s
    mask_k = list_k_s < list_k_l
    n_rho_fix = np.count_nonzero(mask_rho)
    n_cp_fix = np.count_nonzero(mask_cp)
    n_k_fix = np.count_nonzero(mask_k)
    if n_rho_fix or n_cp_fix or n_k_fix:
        print(f"⚠️ 对采样结果做了修正：rho 修正 {n_rho_fix} 个，cp 修正 {n_cp_fix} 个，k 修正 {n_k_fix} 个")
        if n_rho_fix:
            list_rho_s[mask_rho] = np.minimum(list_rho_l[mask_rho] + min_delta_rho, 950.0)
        if n_cp_fix:
            list_cl_l[mask_cp] = np.minimum(list_cp_s[mask_cp] + min_delta_cp, 2600.0)
        if n_k_fix:
            list_k_s[mask_k] = np.minimum(list_k_l[mask_k] + min_delta_k, 0.250)

    print(f"参数生成完毕 (共 {N_SAMPLES} 组).")

    # --- 初始化 CSV ---
    headers = [
        'ID', 
        'Input_rho_s', 'Input_rho_l', 'Input_cp_s', 'Input_cl_l', 
        'Input_k_s', 'Input_k_l', 'Input_Lf', 'Input_wt', 
        'Input_Tm', 'Input_dT', # 共 10 个输入 (无 Vr, 无 T1)
        'Output_Tout', 'Output_LiquidFrac', 
        'Time_Cost_s',
        'TimeSeries_File',  # 新增：时间序列数据文件名
        'Status'  # 新增：状态 (Success/Timeout/Error)
    ]

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
        print(f"✅ 已新建文件: {CSV_FILE}")
    else:
        print(f"⚠️ 文件已存在，将在末尾追加数据: {CSV_FILE}")

    print("="*60)
    print("🚀 开始批量仿真任务 (10 参数版 - 流速固定)")
    print("="*60)

    # --- 遍历循环 ---
    start_time_all = time.time()
    success_count = 0
    fail_count = 0
    timeout_count = 0
    
    # 超时检测标志
    timeout_flag = False

    for i in range(N_SAMPLES):
        iter_start = time.time()
        iter_id = i + 1
        
        # 重置超时标志
        timeout_flag = False
        
        # A. 准备当前组参数
        p_rho_s = list_rho_s[i]
        p_rho_l = list_rho_l[i]
        p_cp_s  = list_cp_s[i]
        p_cl_l  = list_cl_l[i]
        p_k_s   = list_k_s[i]
        p_k_l   = list_k_l[i]
        p_lf    = list_Lf[i]
        p_wt    = list_wt[i]
        p_tm    = list_Tm[i]
        p_dt    = list_dT[i]

        # 为当前迭代生成唯一标识符
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        time_series_filename = f"ts_{iter_id:04d}_{time_stamp}.npz"
        time_series_filepath = os.path.join(TIME_SERIES_DIR, time_series_filename)
        
        print(f"\n[Run {iter_id}/{N_SAMPLES}] {time.strftime('%H:%M:%S')} 开始...")
        print(f"  参数: Tm={p_tm:.1f}, Lf={p_lf:.1f}, k_s={p_k_s:.3f}, wt={p_wt:.3f}")
        
        try:
            # B. 修改模型参数 (共 10 个)
            model.parameter('rho_s', f"{p_rho_s}[kg/m^3]")
            model.parameter('rho_l', f"{p_rho_l}[kg/m^3]")
            model.parameter('cp_s',  f"{p_cp_s}[J/(kg*K)]")
            model.parameter('cl_l',  f"{p_cl_l}[J/(kg*K)]") 
            model.parameter('k_s',   f"{p_k_s}[W/(m*K)]")
            model.parameter('k_l',   f"{p_k_l}[W/(m*K)]")
            model.parameter('Lf',    f"{p_lf}[kJ/kg]")
            model.parameter('wt',    f"{p_wt}")
            model.parameter('Tm',    f"{p_tm}[degC]")
            model.parameter('dT',    f"{p_dt}[K]")

            # C. 重置与求解（带简单超时检测）
            model.reset()
            print("  [求解中] ... ", end="", flush=True)
            solve_start = time.time()
            
            # 使用子进程求解以支持可靠超时
            csv_ts_filepath = time_series_filepath.replace('.npz', '.csv')
            params_dict = {
                'rho_s': f"{p_rho_s}[kg/m^3]",
                'rho_l': f"{p_rho_l}[kg/m^3]",
                'cp_s':  f"{p_cp_s}[J/(kg*K)]",
                'cl_l':  f"{p_cl_l}[J/(kg*K)]",
                'k_s':   f"{p_k_s}[W/(m*K)]",
                'k_l':   f"{p_k_l}[W/(m*K)]",
                'Lf':    f"{p_lf}[kJ/kg]",
                'wt':    f"{p_wt}",
                'Tm':    f"{p_tm}[degC]",
                'dT':    f"{p_dt}[K]"
            }
            if USE_SUBPROCESS_SOLVE:
                result = solve_with_timeout("DeepOnet.mph", params_dict, time_series_filepath, csv_ts_filepath, TIMEOUT_SECONDS)
            else:
                # 保留兼容的线程实现
                import threading
                result_container = {'result': None, 'error': None, 'finished': False}
                def solve_in_thread():
                    try:
                        model.solve('研究 1')
                        result_container['result'] = 'success'
                        result_container['finished'] = True
                    except Exception as e:
                        result_container['error'] = e
                        result_container['finished'] = True
                solve_thread = threading.Thread(target=solve_in_thread)
                solve_thread.daemon = True
                solve_thread.start()
                elapsed_time = 0
                check_interval = 1
                while not result_container['finished'] and elapsed_time < TIMEOUT_SECONDS:
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                    if elapsed_time % 30 == 0:
                        print(f"[{elapsed_time//60:.0f}:{elapsed_time%60:02.0f}]", end="", flush=True)
                if not result_container['finished']:
                    result = {'status': 'timeout'}
                elif result_container['error'] is not None:
                    result = {'status': 'error', 'error': str(result_container['error'])}
                else:
                    # 线程成功但我们需要从主模型获取结果（保持与旧逻辑兼容）
                    txt_path = time_series_filepath.replace('.npz', '.txt')
                    try:
                        times, temps = get_T2(model, ts_txt_path=txt_path, ts_csv_path=csv_ts_filepath)
                    except Exception as e:
                        raise
                    tout_val = temps[-1]
                    liquid_raw = model.evaluate('aveop1(ht.alpha12)', inner='last')
                    if hasattr(liquid_raw, 'item'):
                        liquid_val = float(liquid_raw.item())
                    elif isinstance(liquid_raw, (list, np.ndarray)):
                        liquid_val = float(np.array(liquid_raw).flat[0])
                    else:
                        liquid_val = float(liquid_raw)
                    # 保存时间序列
                    np.savez(time_series_filepath, times=times, temperatures=temps, parameters={'id': iter_id, 'rho_s': p_rho_s, 'rho_l': p_rho_l, 'cp_s': p_cp_s, 'cl_l': p_cl_l, 'k_s': p_k_s, 'k_l': p_k_l, 'Lf': p_lf, 'wt': p_wt, 'Tm': p_tm, 'dT': p_dt}, results={'final_temp': tout_val, 'liquid_fraction': liquid_val})
                    outlet_csv = time_series_filepath.replace('.npz', '.outlet.csv')
                    pd.DataFrame({'Time': times, 'Temperature': temps}).to_csv(outlet_csv, index=False)
                    result = {'status': 'success', 'tout': float(tout_val), 'liquid': float(liquid_val)}
            
            # 处理子进程/线程结果
            status = result.get('status')
            if status == 'timeout':
                timeout_flag = True
                iter_time = time.time() - iter_start
                timeout_count += 1

                print(f"\n⏰ 超时警告！求解时间超过{TIMEOUT_SECONDS/60:.1f}分钟")
                print(f"  超时参数: ID={iter_id}, Tm={p_tm:.1f}, Lf={p_lf:.1f}")
                print(f"  实际运行时间: {iter_time:.2f}s")

                # 记录超时信息到CSV
                row = [
                    iter_id,
                    p_rho_s, p_rho_l, p_cp_s, p_cl_l,
                    p_k_s, p_k_l, p_lf, p_wt,
                    p_tm, p_dt,
                    None, None,  # 输出结果为空
                    iter_time,
                    None,
                    'Timeout'
                ]
                with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(row)

                # 记录超时参数到日志
                timeout_log = os.path.join(TIME_SERIES_DIR, f"timeout_{iter_id:04d}_{time_stamp}.txt")
                with open(timeout_log, 'w', encoding='utf-8') as f:
                    f.write(f"ID: {iter_id}\n")
                    f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Status: Timeout (>{TIMEOUT_SECONDS/60:.1f}min)\n")
                    f.write(f"Actual Time: {iter_time:.2f}s\n")
                    f.write(f"Parameters:\n")
                    f.write(f"  rho_s={p_rho_s}, rho_l={p_rho_l}\n")
                    f.write(f"  cp_s={p_cp_s}, cl_l={p_cl_l}\n")
                    f.write(f"  k_s={p_k_s}, k_l={p_k_l}\n")
                    f.write(f"  Lf={p_lf}, wt={p_wt}\n")
                    f.write(f"  Tm={p_tm}, dT={p_dt}\n")

                try:
                    model.reset()
                    print("  已尝试重置模型状态...")
                except:
                    print("  模型重置可能失败，将继续尝试...")

                continue

            elif status == 'error':
                raise RuntimeError(result.get('error', '子进程/线程返回错误'))

            else:
                # 成功。子进程会已保存时间序列文件；取得最终数值
                tout_val = result.get('tout')
                liquid_val = result.get('liquid')
                solve_time = time.time() - solve_start
                print(f"完成! (求解耗时: {solve_time:.2f}s)")

            # E. 保存/加载时间序列数据到文件
            # 如果子进程已经保存了时间序列文件，则直接加载它以避免未定义变量错误
            csv_ts_filepath = time_series_filepath.replace('.npz', '.csv')
            if os.path.exists(time_series_filepath):
                try:
                    with np.load(time_series_filepath) as data:
                        times = data['times']
                        temps = data['temperatures']
                except Exception:
                    # 加载失败则置为空数组以防报错
                    times = np.array([])
                    temps = np.array([])
            else:
                # 回退：尝试从当前模型读取（如果可用），否则保存空数组
                try:
                    txt_path = time_series_filepath.replace('.npz', '.txt')
                    times, temps = get_T2(model, ts_txt_path=txt_path, ts_csv_path=csv_ts_filepath)
                except Exception:
                    times = np.array([])
                    temps = np.array([])
                # 保存时间序列（避免子进程未保存的情况）
                try:
                    np.savez(
                        time_series_filepath,
                        times=times,
                        temperatures=temps,
                        parameters={
                            'id': iter_id,
                            'rho_s': p_rho_s,
                            'rho_l': p_rho_l,
                            'cp_s': p_cp_s,
                            'cl_l': p_cl_l,
                            'k_s': p_k_s,
                            'k_l': p_k_l,
                            'Lf': p_lf,
                            'wt': p_wt,
                            'Tm': p_tm,
                            'dT': p_dt
                        },
                        results={
                            'final_temp': tout_val,
                            'liquid_fraction': liquid_val
                        }
                    )
                    pd.DataFrame({'Time': times, 'Temperature': temps}).to_csv(csv_ts_filepath, index=False)
                except Exception:
                    pass
            
            # 确保 CSV 文件存在（如果子进程已生成则跳过覆盖）
            if not os.path.exists(csv_ts_filepath):
                try:
                    pd.DataFrame({'Time': times, 'Temperature': temps}).to_csv(csv_ts_filepath, index=False)
                except Exception:
                    pass

            # F. 记录迭代时间信息
            iter_time = time.time() - iter_start
            
            print(f"  [结果] Tout: {tout_val:.4f} | LiqFrac: {liquid_val:.4f}")
            print(f"  [时间] 本次迭代总耗时: {iter_time:.2f}s | 累计运行时间: {(time.time()-start_time_all)/60:.2f}min")
            print(f"  [存储] 时间序列已保存到: {time_series_filename}")

            # G. 保存到主CSV
            row = [
                iter_id, 
                p_rho_s, p_rho_l, p_cp_s, p_cl_l, 
                p_k_s, p_k_l, p_lf, p_wt, 
                p_tm, p_dt, 
                tout_val, liquid_val, 
                iter_time,
                time_series_filename,  # 存储文件名，便于查找
                'Success'  # 成功标记
            ]
            
            with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
            
            success_count += 1

        except Exception as e:
            fail_count += 1
            iter_time = time.time() - iter_start
            
            print(f"\n❌ 出错: {e}")
            print(f"  参数: ID={iter_id}, Tm={p_tm:.1f}, Lf={p_lf:.1f}")
            print(f"  本次运行耗时: {iter_time:.2f}s")
            
            # 记录失败的参数
            error_log = os.path.join(TIME_SERIES_DIR, f"error_{iter_id:04d}_{time_stamp}.txt")
            with open(error_log, 'w', encoding='utf-8') as f:
                f.write(f"ID: {iter_id}\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: Error\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Parameters:\n")
                f.write(f"  rho_s={p_rho_s}, rho_l={p_rho_l}\n")
                f.write(f"  cp_s={p_cp_s}, cl_l={p_cl_l}\n")
                f.write(f"  k_s={p_k_s}, k_l={p_k_l}\n")
                f.write(f"  Lf={p_lf}, wt={p_wt}\n")
                f.write(f"  Tm={p_tm}, dT={p_dt}\n")
            
            # 记录到CSV
            status = 'Error' if not timeout_flag else 'Timeout'
            row = [
                iter_id, 
                p_rho_s, p_rho_l, p_cp_s, p_cl_l, 
                p_k_s, p_k_l, p_lf, p_wt, 
                p_tm, p_dt, 
                None, None,  # 输出结果为空
                iter_time,
                None,  # 没有时间序列文件
                status  # 状态标记
            ]
            
            with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
            
            # 尝试重置模型
            try:
                model.reset()
            except:
                pass
            
            continue

    # --- 结束 ---
    total_time_min = (time.time() - start_time_all) / 60
    print("\n" + "="*60)
    print("✅ 批量仿真完成！")
    print(f"📊 统计: 成功 {success_count}/{N_SAMPLES}, 超时 {timeout_count}/{N_SAMPLES}, 失败 {fail_count}/{N_SAMPLES}")
    print(f"⏱️  总耗时: {total_time_min:.2f} 分钟")
    print(f"📂 数据文件: {os.path.abspath(CSV_FILE)}")
    print(f"📂 时间序列数据目录: {os.path.abspath(TIME_SERIES_DIR)}")
    print("="*60)

if __name__ == "__main__":
    # 在Windows上，确保multiprocessing正确初始化
    from multiprocessing import freeze_support
    freeze_support()
    # 运行主函数
    run_simulation_batch()
