"""
KRW IRS Bootstrapping using QuantLib
- Scenario 1: Piecewise Linear Forward (주어진 만기만 사용)
- Scenario 2: Linear Interpolation of IRS rates for intermediate maturities
- Instantaneous Forward Rate Chart로 비교
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

# Overnight Deposit 금리
overnight_rate = 0.04  

# IRS 금리 데이터 (만기별) - 여기서 수정
irs_rates = {
    1: 0.03,
    2: 0.03,
    3: 0.03,
    4: 0.03,
    5: 0.03
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
    
    # Overnight Deposit Helper 추가 (spot date부터 시작하도록 fixing days = settlement_days)
    overnight_quote = ql.QuoteHandle(ql.SimpleQuote(overnight_rate))
    overnight_helper = ql.DepositRateHelper(
        overnight_quote,
        ql.Period(1, ql.Days),
        settlement_days,  # fixing days = settlement_days로 설정하여 spot date부터 시작
        calendar,
        business_convention,
        False,  # endOfMonth
        day_count
    )
    helpers.append(overnight_helper)
    
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
    """Piecewise Linear Forward로 커브 생성"""
    curve = ql.PiecewiseLinearForward(settlement_days, calendar, helpers, day_count)
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


def get_instantaneous_forward_rates(curve, num_points: int = 100) -> List[Tuple[float, float]]:
    """커브에서 instantaneous forward rate 추출"""
    spot_date = curve.referenceDate()
    rates = []
    
    for i in range(num_points + 1):
        tenor = max_maturity * i / num_points
        target_date = spot_date + ql.Period(int(tenor * 365), ql.Days)
        
        # Instantaneous forward rate (연속복리)
        fwd_rate = curve.forwardRate(
            target_date, target_date, day_count, ql.Continuous
        ).rate() * 100
        
        rates.append((tenor, fwd_rate))
    
    return rates


# ============================================================================
# Step Function Chart
# ============================================================================

class InstantaneousForwardPlotter:
    """여러 시나리오의 Instantaneous Forward Rate을 비교하는 그래프 생성"""
    
    def __init__(self, figsize: Tuple[float, float] = (14, 8)):
        self.figsize = figsize
        self.scenarios = {}
        
    def add_scenario(self, name: str, rates: List[Tuple[float, float]]):
        self.scenarios[name] = rates
        
    def plot(self,
             title: str = "Instantaneous Forward Rate Comparison",
             xlabel: str = "Time (Years)",
             ylabel: str = "Instantaneous Forward Rate (%)",
             colors: Optional[List[str]] = None,
             fill: bool = True,
             fill_alpha: float = 0.15,
             show_grid: bool = True,
             save_path: Optional[str] = None):
        
        if not self.scenarios:
            raise ValueError("No scenarios to plot.")
        
        default_colors = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea']
        colors = colors or default_colors
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for idx, (name, rates) in enumerate(self.scenarios.items()):
            color = colors[idx % len(colors)]
            
            tenors = [t for t, _ in rates]
            fwd_rates = [r for _, r in rates]
            
            # 선 그리기
            ax.plot(tenors, fwd_rates, color=color, linewidth=2.5, label=name)
            
            # 영역 채우기
            if fill:
                ax.fill_between(tenors, 0, fwd_rates, color=color, alpha=fill_alpha)
        
        # 축 설정
        all_rates = []
        for rates in self.scenarios.values():
            all_rates.extend([r for _, r in rates])
        
        y_margin = max(all_rates) * 0.15
        
        ax.set_xlim(0, max_maturity)
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
    print("KRW IRS Bootstrapping (Piecewise Linear Forward)")
    print("=" * 80)
    print(f"\n기준일: {today}")
    print(f"\nOvernight Deposit 금리: {overnight_rate*100:.2f}%")
    print(f"\n입력 IRS 금리:")
    for years, rate in sorted(irs_rates.items()):
        print(f"  {years}년: {rate*100:.2f}%")
    
    # 만기 리스트
    num_quarters = int(max_maturity * 4)
    maturities = [i * 0.25 for i in range(1, num_quarters + 1)]
    
    # 커브 생성
    print("\n" + "-" * 80)
    print("Piecewise Linear Forward 커브 생성")
    print("-" * 80)
    
    helpers = create_irs_helpers(use_interpolation=False)
    curve = build_curve(helpers)
    results = calculate_quarterly_zero_rates(curve, maturities)
    fwd_rates = get_instantaneous_forward_rates(curve)
    
    # Zero Rate 출력
    print("\n" + "=" * 80)
    print("Quarterly Compounding Zero Rate")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print("\n")
    print(df.to_string(index=False, float_format='%.6f'))
    
    # Instantaneous Forward Rate Chart 그리기
    print("\n" + "=" * 80)
    print("Instantaneous Forward Rate Chart 생성 중...")
    print("=" * 80)
    
    plotter = InstantaneousForwardPlotter(figsize=(16, 8))
    plotter.add_scenario("Piecewise Linear Forward", fwd_rates)
    
    plotter.plot(
        title=f"Instantaneous Forward Rate (O/N={overnight_rate*100:.0f}%, IRS: {', '.join([f'{y}Y={r*100:.0f}%' for y, r in sorted(irs_rates.items())])})",
        xlabel="Time (Years)",
        ylabel="Instantaneous Forward Rate (%)",
        colors=['#2563eb'],
        fill=True,
        fill_alpha=0.15,
        save_path="forward_rate_comparison.png"
    )


if __name__ == "__main__":
    main()


