"""
KRW IRS Bootstrapping using QuantLib
- Scenario 1: Piecewise Constant Instantaneous Forward (주어진 만기만 사용)
- Scenario 2: Linear Interpolation of IRS rates for intermediate maturities
- Step Function Chart로 Forward Rate 비교
"""

import QuantLib as ql
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional

# ============================================================================
# 입력 데이터 설정
# ============================================================================

# 기준일
today = ql.Date(17, 12, 2024)
ql.Settings.instance().evaluationDate = today

# KRW 캘린더 및 컨벤션
calendar = ql.SouthKorea()
day_count = ql.Actual365Fixed()
business_convention = ql.ModifiedFollowing
settlement_days = 2

# IRS 금리 데이터 (만기별) - 여기서 수정
irs_rates = {
    1: 0.02,
    2: 0.04,
    3: 0.05
}

# 최대 만기 (자동 계산)
max_maturity = max(irs_rates.keys())


# ============================================================================
# 유틸리티 함수
# ============================================================================

def linear_interpolate(x, x1, y1, x2, y2):
    """선형 보간"""
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def get_interpolated_rate(tenor: float) -> float:
    """주어진 tenor에 대한 보간된 금리 반환"""
    sorted_maturities = sorted(irs_rates.keys())
    
    # 첫 번째 만기 이하면 첫 번째 금리 사용
    if tenor <= sorted_maturities[0]:
        return irs_rates[sorted_maturities[0]]
    
    # 마지막 만기 이상이면 마지막 금리 사용
    if tenor >= sorted_maturities[-1]:
        return irs_rates[sorted_maturities[-1]]
    
    # 중간 만기: 선형 보간
    for i in range(len(sorted_maturities) - 1):
        if sorted_maturities[i] <= tenor <= sorted_maturities[i + 1]:
            return linear_interpolate(
                tenor,
                sorted_maturities[i], irs_rates[sorted_maturities[i]],
                sorted_maturities[i + 1], irs_rates[sorted_maturities[i + 1]]
            )
    
    return irs_rates[sorted_maturities[-1]]


# ============================================================================
# QuantLib 부트스트래핑
# ============================================================================

def create_irs_helpers(use_interpolation=False):
    """
    IRS Helper 생성
    use_interpolation: True면 중간 만기에 선형 보간된 금리 사용
    """
    helpers = []
    
    if use_interpolation:
        # Scenario 2: 선형 보간 사용 - 0.25년 간격으로 모든 만기 생성
        num_quarters = int(max_maturity * 4)
        
        for i in range(1, num_quarters + 1):
            tenor = i * 0.25
            rate = get_interpolated_rate(tenor)
            
            months = int(tenor * 12)
            swap_rate = ql.QuoteHandle(ql.SimpleQuote(rate))
            
            helper = ql.SwapRateHelper(
                swap_rate,
                ql.Period(months, ql.Months),
                calendar,
                ql.Quarterly,
                business_convention,
                day_count,
                ql.Euribor3M()
            )
            helpers.append(helper)
    else:
        # Scenario 1: 주어진 만기만 사용
        for years, rate in sorted(irs_rates.items()):
            swap_rate = ql.QuoteHandle(ql.SimpleQuote(rate))
            
            helper = ql.SwapRateHelper(
                swap_rate,
                ql.Period(years, ql.Years),
                calendar,
                ql.Quarterly,
                business_convention,
                day_count,
                ql.Euribor3M()
            )
            helpers.append(helper)
    
    return helpers


def build_curve(helpers):
    """Piecewise Flat Forward로 커브 생성"""
    curve = ql.PiecewiseFlatForward(settlement_days, calendar, helpers, day_count)
    curve.enableExtrapolation()
    return curve


def calculate_quarterly_zero_rates(curve, maturities):
    """Quarterly Compounding Zero Rate 계산"""
    spot_date = curve.referenceDate()
    results = []
    
    for tenor in maturities:
        target_date = spot_date + ql.Period(int(tenor * 12), ql.Months)
        
        zero_rate = curve.zeroRate(
            target_date,
            day_count,
            ql.Compounded,
            ql.Quarterly
        ).rate()
        
        results.append({
            'Tenor (Years)': tenor,
            'Zero Rate (%)': zero_rate * 100
        })
    
    return results


def get_forward_rate_intervals(curve) -> List[Tuple[float, float, float]]:
    """커브에서 3개월 forward rate 구간 데이터 추출"""
    spot_date = curve.referenceDate()
    intervals = []
    num_quarters = int(max_maturity * 4)
    
    for i in range(num_quarters):
        start_tenor = i * 0.25
        end_tenor = (i + 1) * 0.25
        
        start_date = spot_date + ql.Period(int(start_tenor * 12), ql.Months)
        end_date = spot_date + ql.Period(int(end_tenor * 12), ql.Months)
        
        fwd_rate = curve.forwardRate(
            start_date, end_date, day_count, ql.Compounded, ql.Quarterly
        ).rate() * 100
        
        intervals.append((start_tenor, end_tenor, fwd_rate))
    
    return intervals


# ============================================================================
# Step Function Chart
# ============================================================================

class MultiStepFunctionPlotter:
    """여러 시나리오의 Step Function을 비교하는 그래프 생성"""
    
    def __init__(self, figsize: Tuple[float, float] = (14, 8)):
        self.figsize = figsize
        self.scenarios = {}
        
    def add_scenario(self, name: str, intervals: List[Tuple[float, float, float]]):
        self.scenarios[name] = intervals
        
    def plot(self,
             title: str = "Forward Rate Comparison",
             xlabel: str = "Time (Years)",
             ylabel: str = "Forward Rate (%)",
             colors: Optional[List[str]] = None,
             show_values: bool = False,
             fill: bool = True,
             fill_alpha: float = 0.15,
             show_grid: bool = True,
             save_path: Optional[str] = None):
        
        if not self.scenarios:
            raise ValueError("No scenarios to plot.")
        
        default_colors = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea']
        colors = colors or default_colors
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for idx, (name, intervals) in enumerate(self.scenarios.items()):
            color = colors[idx % len(colors)]
            sorted_data = sorted(intervals, key=lambda x: x[0])
            
            # 수평선 그리기
            for t_start, t_end, rate in sorted_data:
                ax.hlines(y=rate, xmin=t_start, xmax=t_end, 
                         colors=color, linewidth=2.5, 
                         label=name if t_start == sorted_data[0][0] else None)
            
            # 수직선 그리기
            for i in range(len(sorted_data) - 1):
                _, t_end, rate1 = sorted_data[i]
                t_start_next, _, rate2 = sorted_data[i + 1]
                if abs(t_end - t_start_next) < 1e-10:
                    ax.vlines(x=t_end, ymin=min(rate1, rate2), ymax=max(rate1, rate2),
                             colors=color, linewidth=2.5, linestyle='--', alpha=0.7)
            
            # 영역 채우기
            if fill:
                for t_start, t_end, rate in sorted_data:
                    rect = patches.Rectangle(
                        (t_start, 0), t_end - t_start, rate,
                        linewidth=0, facecolor=color, alpha=fill_alpha
                    )
                    ax.add_patch(rect)
        
        # 축 설정
        all_times = []
        all_rates = []
        for intervals in self.scenarios.values():
            for t_start, t_end, rate in intervals:
                all_times.extend([t_start, t_end])
                all_rates.append(rate)
        
        x_margin = (max(all_times) - min(all_times)) * 0.05
        y_margin = max(all_rates) * 0.15
        
        ax.set_xlim(min(all_times) - x_margin, max(all_times) + x_margin)
        ax.set_ylim(0, max(all_rates) + y_margin)
        
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_axisbelow(True)
        
        # x축 눈금
        num_ticks = int(max_maturity * 4) + 1
        ax.set_xticks([i * 0.25 for i in range(num_ticks)])
        ax.set_xticklabels([f'{i * 0.25:.2f}' for i in range(num_ticks)], rotation=45)
        
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Graph saved to: {save_path}")
        
        plt.show()
        return fig, ax


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("=" * 80)
    print("KRW IRS Bootstrapping 비교")
    print("=" * 80)
    print(f"\n기준일: {today}")
    print(f"\n입력 IRS 금리:")
    for years, rate in sorted(irs_rates.items()):
        print(f"  {years}년: {rate*100:.2f}%")
    
    # 만기 리스트
    num_quarters = int(max_maturity * 4)
    maturities = [i * 0.25 for i in range(1, num_quarters + 1)]
    
    # Scenario 1: 주어진 만기만 사용
    print("\n" + "-" * 80)
    print("Scenario 1: Piecewise Constant Instantaneous Forward (주어진 만기만 사용)")
    print("-" * 80)
    
    helpers1 = create_irs_helpers(use_interpolation=False)
    curve1 = build_curve(helpers1)
    results1 = calculate_quarterly_zero_rates(curve1, maturities)
    intervals1 = get_forward_rate_intervals(curve1)
    
    # Scenario 2: 선형 보간 사용
    print("\n" + "-" * 80)
    print("Scenario 2: 선형 보간된 IRS 금리 사용")
    print("-" * 80)
    
    # 보간된 금리 출력
    print("\n보간된 IRS 금리:")
    for i in range(1, num_quarters + 1):
        tenor = i * 0.25
        rate = get_interpolated_rate(tenor)
        print(f"  {tenor:.2f}년: {rate*100:.4f}%")
    
    helpers2 = create_irs_helpers(use_interpolation=True)
    curve2 = build_curve(helpers2)
    results2 = calculate_quarterly_zero_rates(curve2, maturities)
    intervals2 = get_forward_rate_intervals(curve2)
    
    # Zero Rate 비교
    print("\n" + "=" * 80)
    print("Quarterly Compounding Zero Rate 비교")
    print("=" * 80)
    
    df1 = pd.DataFrame(results1)
    df2 = pd.DataFrame(results2)
    
    comparison = pd.DataFrame({
        'Tenor (Years)': df1['Tenor (Years)'],
        'Scenario 1 (%)': df1['Zero Rate (%)'],
        'Scenario 2 (%)': df2['Zero Rate (%)'],
        'Difference (bp)': (df1['Zero Rate (%)'] - df2['Zero Rate (%)']) * 100
    })
    
    print("\n")
    print(comparison.to_string(index=False, float_format='%.6f'))
    
    # Forward Rate 비교
    print("\n" + "=" * 80)
    print("3개월 Forward Rate 비교")
    print("=" * 80)
    
    fwd_comparison = pd.DataFrame({
        'Period': [f'{s:.2f}Y - {e:.2f}Y' for s, e, _ in intervals1],
        'Scenario 1 (%)': [r for _, _, r in intervals1],
        'Scenario 2 (%)': [r for _, _, r in intervals2],
        'Difference (bp)': [(r1 - r2) * 100 for (_, _, r1), (_, _, r2) in zip(intervals1, intervals2)]
    })
    
    print("\n")
    print(fwd_comparison.to_string(index=False, float_format='%.6f'))
    
    # Step Function Chart 그리기
    print("\n" + "=" * 80)
    print("Step Function Chart 생성 중...")
    print("=" * 80)
    
    plotter = MultiStepFunctionPlotter(figsize=(16, 8))
    plotter.add_scenario("Scenario 1: Piecewise Constant (원본 만기만)", intervals1)
    plotter.add_scenario("Scenario 2: Linear Interpolation (보간)", intervals2)
    
    plotter.plot(
        title=f"3-Month Forward Rate Comparison (IRS: {', '.join([f'{y}Y={r*100:.0f}%' for y, r in sorted(irs_rates.items())])})",
        xlabel="Time (Years)",
        ylabel="3M Forward Rate (%)",
        colors=['#2563eb', '#dc2626'],
        fill=True,
        fill_alpha=0.15,
        save_path="forward_rate_comparison.png"
    )


if __name__ == "__main__":
    main()


