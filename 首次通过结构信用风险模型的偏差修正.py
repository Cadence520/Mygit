"""
A股信用风险参数估计 - 完整流程整合版
基于 Amaya-Deng-2019 论文复现
整合四个步骤：数据准备 → 迭代估计 → NMLE/CMLE → DebCMLE
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.integrate import quad
import json
import warnings
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import os

warnings.filterwarnings('ignore')

# ==================== 全局配置参数 ====================
class Config:
    """全局配置类"""
    # 路径配置
    DATA_PATH = r"./credit_risk_data"  # 修改为当前目录下的子文件夹

    # 时间参数
    ESTIMATION_START = "2014-01-01"
    ESTIMATION_END = "2016-12-31"
    DATA_START = "2013-01-01"
    LOOKBACK_MONTHS = 12
    MIN_TRADING_DAYS = 200

    # 迭代参数
    MAX_ITER = 100
    TOLERANCE = 1e-4

    # DebCMLE参数
    NU_GRID_SIZE = 60  # 降低以加快速度
    Z_INTEGRATION_POINTS = 300

    # 并行参数
    N_PROCESSES = max(1, cpu_count() - 1)

    # 输出文件名
    OUTPUT_STEP1 = "step1_样本数据.xlsx"
    OUTPUT_STEP2 = "step2_迭代估计.xlsx"
    OUTPUT_STEP3 = "step3_MLE估计.xlsx"
    OUTPUT_STEP4 = "step4_DebCMLE最终结果.xlsx"
    OUTPUT_REPORT = "完整质量报告.xlsx"

# ==================== 工具函数 ====================
def ensure_directory(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"  ✓ 已创建目录: {path}")

def print_section(title, char="=", width=80):
    """打印分隔标题"""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)

def print_progress(current, total, step=500):
    """打印进度"""
    if current % step == 0 or current == total:
        print(f"  进度: {current}/{total} ({100*current/total:.1f}%)")

# ==================== 步骤1: 数据准备 ====================
class DataPreparation:
    """数据准备模块"""

    @staticmethod
    def load_raw_data(data_path):
        """加载原始数据"""
        print("\n【步骤1.1】加载原始数据...")
        try:
            df_financial = pd.read_excel(f"{data_path}/FS_Combas.xlsx")
            df_trading = pd.read_excel(f"{data_path}/TRD_Dalyr.xlsx")
            df_rf = pd.read_excel(f"{data_path}/无风险利率.xlsx")

            print(f"  ✓ 财务数据: {len(df_financial):,} 行")
            print(f"  ✓ 交易数据: {len(df_trading):,} 行")
            print(f"  ✓ 利率数据: {len(df_rf):,} 行")
            return df_financial, df_trading, df_rf
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            print("\n提示: 请确保以下文件存在于 DATA_PATH 目录:")
            print("  - FS_Combas.xlsx (财务数据)")
            print("  - TRD_Dalyr.xlsx (交易数据)")
            print("  - 无风险利率.xlsx")
            raise

    @staticmethod
    def preprocess_data(df_financial, df_trading, df_rf):
        """预处理数据"""
        print("\n【步骤1.2】数据预处理...")

        # 财务数据
        df_financial.columns = ['证券代码', '证券简称', '统计截止日期', '报表类型',
                                 '流动负债', '非流动负债']
        df_financial['统计截止日期'] = pd.to_datetime(df_financial['统计截止日期'], errors='coerce')
        df_financial['流动负债'] = pd.to_numeric(df_financial['流动负债'], errors='coerce')
        df_financial['非流动负债'] = pd.to_numeric(df_financial['非流动负债'], errors='coerce')
        df_financial = df_financial.dropna(subset=['证券代码', '统计截止日期'])
        df_financial['负债面值D'] = (df_financial['流动负债'].fillna(0) + 
                                     0.5 * df_financial['非流动负债'].fillna(0))
        df_financial = df_financial[df_financial['负债面值D'] > 0]

        # 交易数据
        df_trading.columns = ['证券代码', '交易日期', '收盘价', '流通市值_千元']
        df_trading['交易日期'] = pd.to_datetime(df_trading['交易日期'], errors='coerce')
        df_trading['市值_元'] = pd.to_numeric(df_trading['流通市值_千元'], errors='coerce') * 1000
        df_trading = df_trading.dropna(subset=['证券代码', '交易日期', '市值_元'])
        df_trading = df_trading[df_trading['市值_元'] > 0]
        df_trading = df_trading[df_trading['交易日期'] >= Config.DATA_START]

        # 无风险利率
        df_rf.columns = ['统计日期', '无风险利率_pct']
        df_rf['统计日期'] = pd.to_datetime(df_rf['统计日期'], errors='coerce')
        df_rf['无风险利率_pct'] = pd.to_numeric(df_rf['无风险利率_pct'], errors='coerce')
        df_rf = df_rf.dropna()
        df_rf['年月'] = df_rf['统计日期'].dt.to_period('M')
        df_rf['r_年化'] = np.log(1 + df_rf['无风险利率_pct'] / 100)
        df_rf = df_rf.drop_duplicates(subset=['年月'], keep='last')
        rf_lookup = df_rf.set_index('年月')['r_年化'].to_dict()

        print(f"  ✓ 有效财务记录: {len(df_financial):,}")
        print(f"  ✓ 有效交易记录: {len(df_trading):,}")
        print(f"  ✓ 覆盖公司: {df_trading['证券代码'].nunique():,} 家")

        return df_financial, df_trading, rf_lookup

    @staticmethod
    def build_samples(df_financial, df_trading, rf_lookup):
        """构建样本数据"""
        print("\n【步骤1.3】构建月末样本...")

        # 获取月末时点
        estimation_data = df_trading[
            (df_trading['交易日期'] >= Config.ESTIMATION_START) &
            (df_trading['交易日期'] <= Config.ESTIMATION_END)
        ]
        df_dates = estimation_data[['交易日期']].drop_duplicates()
        df_dates['年月'] = df_dates['交易日期'].dt.to_period('M')
        month_ends = df_dates.groupby('年月')['交易日期'].max().reset_index()
        month_ends = month_ends['交易日期'].sort_values().tolist()

        companies = estimation_data['证券代码'].unique()
        print(f"  ✓ 月末时点: {len(month_ends)} 个")
        print(f"  ✓ 样本公司: {len(companies):,} 家")

        # 财务数据查询函数
        def get_latest_D(stock_code, ref_date):
            stock_fin = df_financial[
                (df_financial['证券代码'] == stock_code) &
                (df_financial['统计截止日期'] <= ref_date)
            ]
            if len(stock_fin) == 0:
                return np.nan
            return stock_fin.sort_values('统计截止日期', ascending=False).iloc[0]['负债面值D']

        # 构建样本
        results = []
        total = len(companies) * len(month_ends)
        processed = 0

        for company in companies:
            df_firm = df_trading[df_trading['证券代码'] == company].sort_values('交易日期')

            for t_end in month_ends:
                processed += 1
                print_progress(processed, total, step=5000)

                t_start = t_end - relativedelta(months=Config.LOOKBACK_MONTHS)
                df_window = df_firm[
                    (df_firm['交易日期'] >= t_start) &
                    (df_firm['交易日期'] <= t_end)
                ].copy()

                # 质量检验
                n_days = len(df_window)
                quality_flag = "PASS"
                fail_reason = ""

                if n_days < Config.MIN_TRADING_DAYS:
                    quality_flag = "FAIL"
                    fail_reason = f"交易日不足({n_days}天)"
                elif df_window['市值_元'].isna().any():
                    quality_flag = "FAIL"
                    fail_reason = "市值缺失"
                else:
                    D = get_latest_D(company, t_end)
                    if pd.isna(D) or D <= 0:
                        quality_flag = "FAIL"
                        fail_reason = "负债数据无效"
                    else:
                        year_month = t_end.to_period('M')
                        r = rf_lookup.get(year_month, np.nan)
                        if pd.isna(r):
                            quality_flag = "FAIL"
                            fail_reason = "利率缺失"
                        elif df_window['市值_元'].min() < 0.5 * D:
                            quality_flag = "FAIL"
                            fail_reason = "可能违反幸存条件"

                if quality_flag == "PASS":
                    S_0 = df_window['市值_元'].iloc[0]
                    S_T = df_window['市值_元'].iloc[-1]
                    L = D
                    market_values = df_window[['交易日期', '市值_元']].copy()
                    market_values['交易日期'] = market_values['交易日期'].dt.strftime('%Y-%m-%d')
                    mv_json = market_values.to_json(orient='records', force_ascii=False)
                else:
                    S_0 = S_T = L = r = np.nan
                    mv_json = ""

                results.append({
                    '证券代码': company,
                    '月末日期': t_end.strftime('%Y-%m-%d'),
                    '窗口起始': t_start.strftime('%Y-%m-%d'),
                    '交易日数': n_days,
                    'S_0': S_0,
                    'S_T': S_T,
                    'D': D if quality_flag == "PASS" else np.nan,
                    'L': L if quality_flag == "PASS" else np.nan,
                    'r': r if quality_flag == "PASS" else np.nan,
                    '市值序列': mv_json,
                    '质量标记': quality_flag,
                    '失败原因': fail_reason
                })

        df_output = pd.DataFrame(results)
        n_pass = (df_output['质量标记'] == 'PASS').sum()
        print(f"\n  ✓ 通过质检: {n_pass:,}/{len(df_output):,} ({100*n_pass/len(df_output):.1f}%)")

        return df_output

# ==================== 步骤2: 迭代估计 ====================
class IterativeEstimation:
    """迭代估计模块"""

    @staticmethod
    def black_cox_equity(A, sigma, D, L, r, T):
        """Black-Cox股权定价"""
        if T <= 1e-6:
            return max(A - D, 0)

        sqrt_T = np.sqrt(T)
        eta = (r + 0.5 * sigma**2) / (sigma**2 + 1e-10)
        a = (np.log(A/D) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T + 1e-10)
        b = (2*np.log(L) - np.log(A) - np.log(D) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T + 1e-10)

        term1 = A * norm.cdf(a) - D * np.exp(-r*T) * norm.cdf(a - sigma*sqrt_T)
        term2 = (L/A)**(2*eta) * (A * norm.cdf(b) - D * np.exp(-r*T) * norm.cdf(b - sigma*sqrt_T))

        return term1 - term2

    @staticmethod
    def solve_asset_value(S_obs, sigma, D, L, r, T):
        """反解资产价值"""
        def objective(A):
            return IterativeEstimation.black_cox_equity(A, sigma, D, L, r, T) - S_obs

        try:
            A_lower = L * 1.01
            A_upper = max(10 * S_obs, 100 * D)
            A_solution = brentq(objective, A_lower, A_upper, xtol=1e-6, maxiter=100)
            return A_solution
        except:
            return np.nan

    @staticmethod
    def iterate_single(S_series, D, L, r):
        """单样本迭代估计"""
        n_days = len(S_series)
        returns = np.log(S_series[1:] / S_series[:-1])
        returns = returns[np.isfinite(returns)]

        if len(returns) < 10:
            return np.nan, np.nan, np.nan, 0, False, '收益率不足'

        sigma_k = np.std(returns) * np.sqrt(252)
        if sigma_k <= 0 or not np.isfinite(sigma_k):
            return np.nan, np.nan, np.nan, 0, False, '初始波动率无效'

        for iteration in range(Config.MAX_ITER):
            A_k = np.zeros(n_days)
            for i in range(n_days):
                T_i = 1.0 - (i / (n_days - 1))
                A_k[i] = IterativeEstimation.solve_asset_value(S_series[i], sigma_k, D, L, r, T_i)

            if np.isnan(A_k).any() or (A_k <= 0).any():
                return np.nan, np.nan, np.nan, iteration, False, '资产求解失败'

            if (A_k <= L).any():
                return np.nan, np.nan, np.nan, iteration, False, '违反幸存条件'

            returns_A = np.log(A_k[1:] / A_k[:-1])
            returns_A = returns_A[np.isfinite(returns_A)]

            if len(returns_A) < 10:
                return np.nan, np.nan, np.nan, iteration, False, '资产收益率不足'

            sigma_new = np.std(returns_A) * np.sqrt(252)
            if not np.isfinite(sigma_new) or sigma_new <= 0:
                return np.nan, np.nan, np.nan, iteration, False, '波动率更新失败'

            if abs(sigma_new - sigma_k) < Config.TOLERANCE:
                return sigma_new, A_k[0], A_k[-1], iteration+1, True, ''

            sigma_k = sigma_new

        return sigma_k, A_k[0], A_k[-1], Config.MAX_ITER, False, '未收敛'

    @staticmethod
    def process_sample(idx, row, progress_dict):
        """处理单个样本（多进程）"""
        try:
            daily_data = json.loads(row['市值序列'])
            S_series = pd.DataFrame(daily_data)['市值_元'].values

            sigma, A_0, A_T, n_iter, converged, fail = \
                IterativeEstimation.iterate_single(S_series, row['D'], row['L'], row['r'])

            with progress_dict['lock']:
                progress_dict['count'] += 1
                print_progress(progress_dict['count'], progress_dict['total'], step=100)

            return {
                '证券代码': row['证券代码'],
                '月末日期': row['月末日期'],
                'sigma': sigma,
                'A_0': A_0,
                'A_T': A_T,
                '迭代次数': n_iter,
                '是否收敛': converged,
                '失败原因': fail
            }
        except Exception as e:
            with progress_dict['lock']:
                progress_dict['count'] += 1
            return {
                '证券代码': row.get('证券代码', 'UNKNOWN'),
                '月末日期': row.get('月末日期', 'UNKNOWN'),
                'sigma': np.nan, 'A_0': np.nan, 'A_T': np.nan,
                '迭代次数': 0, '是否收敛': False, '失败原因': f'异常:{str(e)[:50]}'
            }

# ==================== 步骤3: NMLE和CMLE ====================
class MLEEstimation:
    """MLE估计模块"""

    @staticmethod
    def estimate_nmle(z_0, z_T, sigma, T=1.0):
        """NMLE估计（论文公式2.1）"""
        nu_N = (z_T - z_0) / T
        mu_N = nu_N + 0.5 * sigma**2
        return mu_N

    @staticmethod
    def compute_cond_expectation(nu, z_0, sigma, T=1.0):
        """条件期望（附录A.4）"""
        sqrt_T_sigma = sigma * np.sqrt(T)
        y = (z_0 + nu * T) / sqrt_T_sigma

        if y > 10:
            return z_0 + nu*T + sqrt_T_sigma / y
        elif y < -10:
            return np.nan

        pdf_y = norm.pdf(y)
        cdf_y = norm.cdf(y)
        if cdf_y < 1e-15:
            return np.nan

        return (z_0 + nu*T) + sqrt_T_sigma * (pdf_y / cdf_y)

    @staticmethod
    def estimate_cmle(z_0, z_T, sigma, T=1.0):
        """CMLE估计"""
        def objective(nu):
            E_cond = MLEEstimation.compute_cond_expectation(nu, z_0, sigma, T)
            if np.isnan(E_cond):
                return np.inf
            return E_cond - z_T

        try:
            nu_lower, nu_upper = -3.0, 3.0
            obj_lower = objective(nu_lower)
            obj_upper = objective(nu_upper)

            if np.isinf(obj_lower) or np.isinf(obj_upper):
                return np.nan

            if obj_lower * obj_upper > 0:
                nu_lower, nu_upper = -5.0, 5.0
                obj_lower = objective(nu_lower)
                obj_upper = objective(nu_upper)
                if obj_lower * obj_upper > 0:
                    return np.nan

            nu_C = brentq(objective, nu_lower, nu_upper, xtol=1e-6, maxiter=50)
            mu_C = nu_C + 0.5 * sigma**2

            if abs(mu_C) > 5.0:
                return np.nan

            return mu_C
        except:
            return np.nan

    @staticmethod
    def process_sample(idx, row, progress_dict):
        """处理单样本"""
        try:
            z_0 = np.log(row['A_0'] / row['L'])
            z_T = np.log(row['A_T'] / row['L'])
            sigma = row['sigma']

            mu_N = MLEEstimation.estimate_nmle(z_0, z_T, sigma)
            mu_C = MLEEstimation.estimate_cmle(z_0, z_T, sigma)

            with progress_dict['lock']:
                progress_dict['count'] += 1
                print_progress(progress_dict['count'], progress_dict['total'], step=200)

            return {
                '证券代码': row['证券代码'],
                '月末日期': row['月末日期'],
                'mu_NMLE': mu_N,
                'mu_CMLE': mu_C,
                'z_0': z_0,
                'z_T': z_T
            }
        except Exception as e:
            with progress_dict['lock']:
                progress_dict['count'] += 1
            return {
                '证券代码': row.get('证券代码', 'UNKNOWN'),
                '月末日期': row.get('月末日期', 'UNKNOWN'),
                'mu_NMLE': np.nan, 'mu_CMLE': np.nan,
                'z_0': np.nan, 'z_T': np.nan
            }

# ==================== 步骤4: DebCMLE ====================
class DebCMLEEstimation:
    """去偏CMLE估计"""

    @staticmethod
    def survival_prob(z_0, nu, sigma, T=1.0):
        """生存概率（公式2.2）"""
        sqrt_T_sigma = sigma * np.sqrt(T)
        term1 = norm.cdf((z_0 + nu*T) / sqrt_T_sigma)
        term2 = np.exp(-2*z_0*nu / (sigma**2 + 1e-10)) * \
                norm.cdf((nu*T - z_0) / sqrt_T_sigma)
        return max(term1 - term2, 1e-15)

    @staticmethod
    def conditional_pdf(z, z_0, nu, sigma, T=1.0):
        """条件PDF（附录A.1）"""
        sqrt_T_sigma = sigma * np.sqrt(T)
        term1 = norm.pdf((z - (z_0 + nu*T)) / sqrt_T_sigma) / sqrt_T_sigma

        exponent = -2*z_0*(z + nu*T) / (sigma**2 + 1e-10)
        if exponent < -50:
            term2 = 0
        else:
            term2 = np.exp(exponent) * \
                    norm.pdf((z + (z_0 - nu*T)) / sqrt_T_sigma) / sqrt_T_sigma

        numerator = term1 - term2
        denominator = DebCMLEEstimation.survival_prob(z_0, nu, sigma, T)
        return numerator / denominator

    @staticmethod
    def inverse_cmle(z, z_0, sigma, T=1.0):
        """反解CMLE"""
        def objective(nu):
            E_cond = MLEEstimation.compute_cond_expectation(nu, z_0, sigma, T)
            if np.isnan(E_cond):
                return np.inf
            return E_cond - z

        try:
            nu_lower, nu_upper = -5.0, 5.0
            obj_lower = objective(nu_lower)
            obj_upper = objective(nu_upper)

            if np.isinf(obj_lower) or np.isinf(obj_upper):
                return np.nan
            if obj_lower * obj_upper > 0:
                return np.nan

            nu_C_z = brentq(objective, nu_lower, nu_upper, xtol=1e-5, maxiter=30)
            return nu_C_z
        except:
            return np.nan

    @staticmethod
    def compute_bias_function(nu_true, z_0, sigma, T=1.0):
        """计算偏差函数g(ν)"""
        z_max = max(5.0, z_0 + nu_true*T + 5*sigma*np.sqrt(T))
        z_grid = np.linspace(1e-6, z_max, Config.Z_INTEGRATION_POINTS)

        integrand = []
        for z in z_grid:
            nu_C_z = DebCMLEEstimation.inverse_cmle(z, z_0, sigma, T)
            if np.isnan(nu_C_z):
                integrand.append(0)
                continue

            pdf_z = DebCMLEEstimation.conditional_pdf(z, z_0, nu_true, sigma, T)
            integrand.append(nu_C_z * pdf_z)

        integrand = np.array(integrand)
        dz = z_grid[1] - z_grid[0]
        g_value = np.trapz(integrand, dx=dz)

        return g_value

    @staticmethod
    def process_sample(idx, row, progress_dict):
        """处理单样本DebCMLE"""
        try:
            z_0 = row['z_0']
            z_T = row['z_T']
            sigma = row['sigma']
            nu_C_obs = row['mu_CMLE'] - 0.5 * sigma**2

            if np.isnan(nu_C_obs):
                with progress_dict['lock']:
                    progress_dict['count'] += 1
                return {
                    '证券代码': row['证券代码'],
                    '月末日期': row['月末日期'],
                    'mu_DebCMLE': np.nan
                }

            # 构建g函数网格
            nu_grid = np.linspace(-3.0, 3.0, Config.NU_GRID_SIZE)
            g_values = []
            for nu_k in nu_grid:
                g_nu = DebCMLEEstimation.compute_bias_function(nu_k, z_0, sigma)
                g_values.append(g_nu)

            g_values = np.array(g_values)
            valid_mask = np.isfinite(g_values)

            if valid_mask.sum() < 10:
                with progress_dict['lock']:
                    progress_dict['count'] += 1
                return {
                    '证券代码': row['证券代码'],
                    '月末日期': row['月末日期'],
                    'mu_DebCMLE': np.nan
                }

            nu_grid_valid = nu_grid[valid_mask]
            g_values_valid = g_values[valid_mask]

            # 反解g^(-1)(nu_C_obs)
            g_min = g_values_valid.min()
            g_max = g_values_valid.max()

            if nu_C_obs < g_min:
                nu_Deb = nu_grid_valid[g_values_valid.argmin()]
            elif nu_C_obs > g_max:
                nu_Deb = nu_grid_valid[g_values_valid.argmax()]
            else:
                idx_closest = np.argmin(np.abs(g_values_valid - nu_C_obs))
                nu_Deb = nu_grid_valid[idx_closest]

            mu_Deb = nu_Deb + 0.5 * sigma**2

            if abs(mu_Deb) > 5.0:
                mu_Deb = np.nan

            with progress_dict['lock']:
                progress_dict['count'] += 1
                print_progress(progress_dict['count'], progress_dict['total'], step=20)

            return {
                '证券代码': row['证券代码'],
                '月末日期': row['月末日期'],
                'mu_DebCMLE': mu_Deb
            }
        except Exception as e:
            with progress_dict['lock']:
                progress_dict['count'] += 1
            return {
                '证券代码': row.get('证券代码', 'UNKNOWN'),
                '月末日期': row.get('月末日期', 'UNKNOWN'),
                'mu_DebCMLE': np.nan
            }

# ==================== 主流程 ====================
class CreditRiskPipeline:
    """完整流程管理器"""

    def __init__(self):
        self.config = Config()
        ensure_directory(self.config.DATA_PATH)

    def run_step1_data_preparation(self):
        """步骤1: 数据准备"""
        print_section("步骤1: 数据准备")

        df_financial, df_trading, df_rf = DataPreparation.load_raw_data(self.config.DATA_PATH)
        df_financial, df_trading, rf_lookup = DataPreparation.preprocess_data(
            df_financial, df_trading, df_rf
        )
        df_samples = DataPreparation.build_samples(df_financial, df_trading, rf_lookup)

        output_path = os.path.join(self.config.DATA_PATH, self.config.OUTPUT_STEP1)
        df_samples.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\n  ✓ 已保存: {output_path}")

        return df_samples

    def run_step2_iteration(self, df_samples):
        """步骤2: 迭代估计"""
        print_section("步骤2: 迭代估计σ和{A_t}")

        df_valid = df_samples[df_samples['质量标记'] == 'PASS'].copy().reset_index(drop=True)
        print(f"\n【步骤2.1】开始迭代估计 (共{len(df_valid):,}个样本)...")

        with Manager() as manager:
            progress_dict = manager.dict()
            progress_dict['count'] = 0
            progress_dict['total'] = len(df_valid)
            progress_dict['lock'] = manager.Lock()

            with Pool(processes=self.config.N_PROCESSES) as pool:
                process_func = partial(IterativeEstimation.process_sample, progress_dict=progress_dict)
                results = list(pool.starmap(
                    process_func,
                    [(idx, row) for idx, row in df_valid.iterrows()]
                ))

        df_results = pd.DataFrame(results)
        n_success = (df_results['是否收敛'] == True).sum()
        print(f"\n  ✓ 收敛成功: {n_success:,}/{len(df_results):,} ({100*n_success/len(df_results):.1f}%)")

        df_output = pd.concat([
            df_valid[['证券代码', '月末日期', '窗口起始', 'S_0', 'S_T', 'D', 'L', 'r']],
            df_results[['sigma', 'A_0', 'A_T', '迭代次数', '是否收敛', '失败原因']]
        ], axis=1)

        output_path = os.path.join(self.config.DATA_PATH, self.config.OUTPUT_STEP2)
        df_output.to_excel(output_path, index=False, engine='openpyxl')
        print(f"  ✓ 已保存: {output_path}")

        return df_output

    def run_step3_mle(self, df_iteration):
        """步骤3: NMLE和CMLE估计"""
        print_section("步骤3: NMLE和CMLE估计")

        df_valid = df_iteration[df_iteration['是否收敛'] == True].copy().reset_index(drop=True)
        print(f"\n【步骤3.1】开始MLE估计 (共{len(df_valid):,}个样本)...")

        with Manager() as manager:
            progress_dict = manager.dict()
            progress_dict['count'] = 0
            progress_dict['total'] = len(df_valid)
            progress_dict['lock'] = manager.Lock()

            with Pool(processes=self.config.N_PROCESSES) as pool:
                process_func = partial(MLEEstimation.process_sample, progress_dict=progress_dict)
                results = list(pool.starmap(
                    process_func,
                    [(idx, row) for idx, row in df_valid.iterrows()]
                ))

        df_mle = pd.DataFrame(results)
        n_nmle = df_mle['mu_NMLE'].notna().sum()
        n_cmle = df_mle['mu_CMLE'].notna().sum()
        print(f"\n  ✓ NMLE有效: {n_nmle:,} | CMLE有效: {n_cmle:,}")

        df_output = pd.concat([
            df_valid[['证券代码', '月末日期', '窗口起始', 'S_0', 'S_T', 
                      'A_0', 'A_T', 'D', 'L', 'sigma', 'r']],
            df_mle[['mu_NMLE', 'mu_CMLE', 'z_0', 'z_T']]
        ], axis=1)

        output_path = os.path.join(self.config.DATA_PATH, self.config.OUTPUT_STEP3)
        df_output.to_excel(output_path, index=False, engine='openpyxl')
        print(f"  ✓ 已保存: {output_path}")

        return df_output

    def run_step4_debcmle(self, df_mle):
        """步骤4: DebCMLE估计"""
        print_section("步骤4: DebCMLE去偏估计")

        df_valid = df_mle[df_mle['mu_CMLE'].notna()].copy().reset_index(drop=True)
        print(f"\n【步骤4.1】开始DebCMLE估计 (共{len(df_valid):,}个样本)...")
        print(f"  ⚠ 警告: 计算量大，预计耗时 {len(df_valid)*0.5/60/self.config.N_PROCESSES:.0f}-{len(df_valid)*2/60/self.config.N_PROCESSES:.0f} 分钟")

        with Manager() as manager:
            progress_dict = manager.dict()
            progress_dict['count'] = 0
            progress_dict['total'] = len(df_valid)
            progress_dict['lock'] = manager.Lock()

            with Pool(processes=self.config.N_PROCESSES) as pool:
                process_func = partial(DebCMLEEstimation.process_sample, progress_dict=progress_dict)
                results = list(pool.starmap(
                    process_func,
                    [(idx, row) for idx, row in df_valid.iterrows()]
                ))

        df_deb = pd.DataFrame(results)
        n_valid = df_deb['mu_DebCMLE'].notna().sum()
        print(f"\n  ✓ DebCMLE有效: {n_valid:,}/{len(df_deb):,}")

        df_output = pd.concat([
            df_valid[['证券代码', '月末日期', '窗口起始', 'S_0', 'S_T',
                      'A_0', 'A_T', 'D', 'L', 'sigma', 'r',
                      'mu_NMLE', 'mu_CMLE', 'z_0', 'z_T']],
            df_deb[['mu_DebCMLE']]
        ], axis=1)

        output_path = os.path.join(self.config.DATA_PATH, self.config.OUTPUT_STEP4)
        df_output.to_excel(output_path, index=False, engine='openpyxl')
        print(f"  ✓ 已保存: {output_path}")

        return df_output

    def generate_final_report(self, df_final):
        """生成最终质量报告"""
        print_section("生成最终质量报告")

        df_all = df_final[
            (df_final['mu_NMLE'].notna()) &
            (df_final['mu_CMLE'].notna()) &
            (df_final['mu_DebCMLE'].notna())
        ]

        if len(df_all) > 0:
            print(f"\n三方法均有效样本: {len(df_all):,}")

            report_data = {
                '方法': ['NMLE', 'CMLE', 'DebCMLE'],
                '有效样本数': [
                    df_final['mu_NMLE'].notna().sum(),
                    df_final['mu_CMLE'].notna().sum(),
                    df_final['mu_DebCMLE'].notna().sum()
                ],
                '均值(%)': [
                    df_all['mu_NMLE'].mean() * 100,
                    df_all['mu_CMLE'].mean() * 100,
                    df_all['mu_DebCMLE'].mean() * 100
                ],
                '中位数(%)': [
                    df_all['mu_NMLE'].median() * 100,
                    df_all['mu_CMLE'].median() * 100,
                    df_all['mu_DebCMLE'].median() * 100
                ],
                '标准差(%)': [
                    df_all['mu_NMLE'].std() * 100,
                    df_all['mu_CMLE'].std() * 100,
                    df_all['mu_DebCMLE'].std() * 100
                ]
            }

            df_report = pd.DataFrame(report_data)

            print("\n" + "="*60)
            print(df_report.to_string(index=False))
            print("="*60)

            diff_N_Deb = (df_all['mu_NMLE'] - df_all['mu_DebCMLE']) * 100
            diff_C_Deb = (df_all['mu_CMLE'] - df_all['mu_DebCMLE']) * 100

            print(f"\n偏差分析:")
            print(f"  NMLE - DebCMLE: 均值={diff_N_Deb.mean():.2f}%, 中位数={diff_N_Deb.median():.2f}%")
            print(f"  CMLE - DebCMLE: 均值={diff_C_Deb.mean():.2f}%, 中位数={diff_C_Deb.median():.2f}%")

            output_path = os.path.join(self.config.DATA_PATH, self.config.OUTPUT_REPORT)
            df_report.to_excel(output_path, index=False, engine='openpyxl')
            print(f"\n  ✓ 质量报告已保存: {output_path}")
        else:
            print("\n  ✗ 无有效样本生成报告")

    def run(self):
        """运行完整流程"""
        print_section("A股信用风险参数估计 - 完整流程", char="*", width=80)
        print(f"\n配置信息:")
        print(f"  - 数据路径: {self.config.DATA_PATH}")
        print(f"  - 估计期间: {self.config.ESTIMATION_START} 至 {self.config.ESTIMATION_END}")
        print(f"  - 回溯窗口: {self.config.LOOKBACK_MONTHS}个月")
        print(f"  - 并行进程: {self.config.N_PROCESSES}")

        try:
            # 步骤1: 数据准备
            df_samples = self.run_step1_data_preparation()

            # 步骤2: 迭代估计
            df_iteration = self.run_step2_iteration(df_samples)

            # 步骤3: MLE估计
            df_mle = self.run_step3_mle(df_iteration)

            # 步骤4: DebCMLE估计
            df_final = self.run_step4_debcmle(df_mle)

            # 生成报告
            self.generate_final_report(df_final)

            print_section("全部流程完成!", char="*", width=80)
            print(f"\n输出文件位置: {self.config.DATA_PATH}")
            print(f"  1. {self.config.OUTPUT_STEP1}")
            print(f"  2. {self.config.OUTPUT_STEP2}")
            print(f"  3. {self.config.OUTPUT_STEP3}")
            print(f"  4. {self.config.OUTPUT_STEP4} (最终结果)")
            print(f"  5. {self.config.OUTPUT_REPORT}")

        except Exception as e:
            print(f"\n✗ 流程执行失败: {e}")
            import traceback
            traceback.print_exc()

# ==================== 程序入口 ====================
if __name__ == '__main__':
    pipeline = CreditRiskPipeline()
    pipeline.run()
