import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title='UAC Care Load Forecast', layout='wide')

@st.cache_data
def load():
    d = pd.read_csv('uac_clean_2025.csv', parse_dates=['date'])
    m = pd.read_csv('metrics.csv')
    fh = pd.read_csv('forecast_hhs.csv', parse_dates=['date'])
    fd = pd.read_csv('forecast_discharges.csv', parse_dates=['date'])
    return d, m, fh, fd
d, metrics, fc_hhs, fc_dis = load()

# ---------- sidebar ----------
st.sidebar.header('Controls')
H = st.sidebar.slider('Forecast horizon (reports, ~5 per week)', 1, 10, 10)
MODEL = st.sidebar.selectbox('Model', ['GradBoost', 'SARIMA', 'HoltWinters', 'MovingAvg', 'Naive'])
CAP = st.sidebar.number_input('Shelter capacity (assumed)', 2000, 4000, 2550, step=50)
SCEN = st.sidebar.slider('Scenario: change in daily transfers into HHS (%)', -50, 200, 0, step=10)

future_dates = pd.bdate_range(d.date.iloc[-1] + pd.Timedelta(days=1), periods=10, freq='C',
                              weekmask='Sun Mon Tue Wed Thu')

def band_from_rmse(target, model, h_arr):
    """80% band = 1.28 x backtest RMSE, interpolated across horizons 1/5/10."""
    r = metrics[(metrics.target == target) & (metrics.model == model)].sort_values('horizon')
    return 1.2816 * np.interp(h_arr, r.horizon, r.RMSE)

@st.cache_data
def forecast(target, model):
    y = d[target].astype(float).reset_index(drop=True); h = np.arange(1, 11)
    if model == 'GradBoost':
        f = (fc_hhs if target == 'hhs_care' else fc_dis)
        return f.forecast.values, f.lower80.values, f.upper80.values
    if model == 'Naive':
        mid = np.repeat(y.iloc[-1], 10)
    elif model == 'MovingAvg':
        mid = np.repeat(y.iloc[-5:].mean(), 10)
    elif model == 'HoltWinters':
        mid = ExponentialSmoothing(y.values, trend='add', damped_trend=True, seasonal='add',
                                   seasonal_periods=5).fit().forecast(10)
    else:
        res = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 1, 5)).fit(disp=False)
        fc = res.get_forecast(10); ci = fc.conf_int(alpha=0.2)
        return fc.predicted_mean.values, ci.iloc[:, 0].values, ci.iloc[:, 1].values
    b = band_from_rmse(target, model, h)
    return mid, mid - b, mid + b

mid, lo, hi = [a[:H] for a in forecast('hhs_care', MODEL)]
dates = future_dates[:H]
# scenario: extra children per report = baseline transfers x change%, accumulating
extra = d.transferred.tail(20).mean() * SCEN / 100 * np.arange(1, H + 1)
scen = mid + extra
sigma = (hi - lo) / (2 * 1.2816)
p_base = (1 - norm.cdf(CAP, mid, sigma)).max() * 100
p_scen = (1 - norm.cdf(CAP, scen, sigma)).max() * 100

# ---------- header + KPIs ----------
st.title('Predictive Forecasting of Care Load & Placement Demand')
st.caption(f'UAC Program · data to {d.date.iloc[-1]:%d %b %Y} · model: {MODEL} · horizon: {H} reports')
best = metrics[(metrics.target == 'hhs_care') & (metrics.horizon == 5) & (metrics.model == MODEL)]
k = st.columns(4)
k[0].metric('Children in HHS care (latest)', f'{int(d.hhs_care.iloc[-1]):,}')
k[1].metric(f'Forecast in {H} reports', f'{mid[-1]:,.0f}', f'{mid[-1] - d.hhs_care.iloc[-1]:+.0f}')
k[2].metric('Capacity breach probability', f'{p_base:.1f}%',
            f'{p_scen - p_base:+.1f} pts under scenario' if SCEN else None, delta_color='inverse')
k[3].metric('Forecast accuracy (1 wk ahead)', f"{100 - best['MAPE%'].iloc[0]:.1f}%")
if p_base >= 30:
    st.error(f'⚠️ Early warning: {p_base:.0f}% chance of exceeding capacity of {CAP:,} within the horizon.')

# ---------- care load chart ----------
st.subheader('Future care load forecast')
hist = d.tail(60)
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=hi, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig.add_trace(go.Scatter(x=dates, y=lo, mode='lines', fill='tonexty', fillcolor='rgba(42,120,214,0.18)',
                         line=dict(width=0), name='80% interval'))
fig.add_trace(go.Scatter(x=hist.date, y=hist.hhs_care, name='Actual', line=dict(color='#333', width=2)))
fig.add_trace(go.Scatter(x=dates, y=mid, mode='lines+markers', name=f'{MODEL} forecast', line=dict(color='#2a78d6', width=2.5)))
if SCEN:
    fig.add_trace(go.Scatter(x=dates, y=scen, name=f'Scenario ({SCEN:+d}% transfers)',
                             line=dict(color='#eb6834', width=2, dash='dash')))
fig.add_hline(y=CAP, line=dict(color='#d03b3b', dash='dot'), annotation_text=f'Capacity {CAP:,}')
fig.update_layout(height=420, hovermode='x unified', margin=dict(t=10, b=10),
                  yaxis_title='Children in HHS care', legend=dict(orientation='h', y=1.08))
st.plotly_chart(fig, width='stretch')

# ---------- discharge panel ----------
c1, c2 = st.columns([3, 2])
with c1:
    st.subheader('Discharge (placement) demand forecast')
    dm, dl, dh = [a[:H] for a in forecast('discharged', MODEL)]
    f2 = go.Figure()
    f2.add_trace(go.Bar(x=d.tail(30).date, y=d.tail(30).discharged, name='Actual', marker_color='#b0aea6'))
    f2.add_trace(go.Bar(x=dates, y=dm, name='Forecast', marker_color='#eb6834',
                        error_y=dict(type='data', symmetric=False, array=dh - dm, arrayminus=dm - np.maximum(dl, 0))))
    f2.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title='Discharges per report',
                     legend=dict(orientation='h', y=1.1))
    st.plotly_chart(f2, width='stretch')
    st.caption(f'Expected discharges over the horizon: **{dm.sum():.0f}** '
               f'vs expected new transfers: **{d.transferred.tail(20).mean() * (1 + SCEN/100) * H:.0f}**')
with c2:
    st.subheader('Model comparison (test-set MAE)')
    tgt = st.radio('Target', ['hhs_care', 'discharged'], horizontal=True)
    tab = metrics[metrics.target == tgt].pivot_table(index='model', columns='horizon', values='MAE').round(1)
    tab.columns = [f'h={c}' for c in tab.columns]
    st.dataframe(tab.style.format('{:.1f}').highlight_min(axis=0, color='#cde2fb'), width='stretch')
    st.caption('Lower is better. h = reports ahead (1 ≈ next day, 5 ≈ 1 week, 10 ≈ 2 weeks).')

with st.expander('Forecast table'):
    st.dataframe(pd.DataFrame({'date': dates.date, 'forecast': mid.round(0), 'lower80': lo.round(0),
                               'upper80': hi.round(0), 'scenario': scen.round(0)}), width='stretch')
