import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pickle import load
import os
import cvxopt as opt
from cvxopt import solvers

st.set_page_config(page_title="Robo Advisor Dashboard", layout="wide")

# --- Caching for performance ---
@st.cache_data
def load_investor_data():
    return pd.read_csv('InputData.csv', index_col=0)

@st.cache_data
def load_asset_data():
    assets = pd.read_csv('SP500Data.csv', index_col=0)
    missing_fractions = assets.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    assets.drop(labels=drop_list, axis=1, inplace=True)
    assets = assets.ffill()
    return assets

@st.cache_data
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'finalized_model.sav')
    return load(open(model_path, 'rb'))

# --- Prediction and Optimization Functions ---
def predict_risk_tolerance(model, X_input):
    return float(model.predict(X_input)[0])

def get_asset_allocation(risk_tolerance, stock_ticker, assets):
    assets_selected = assets.loc[:, stock_ticker]
    if len(stock_ticker) < 2:
        st.warning('Please select at least 2 assets for portfolio optimization.')
        return None, None
    valid_counts = assets_selected.notna().sum()
    min_valid = valid_counts.min()
    if min_valid < 10:
        st.warning('One or more selected assets have too few valid data points (<10). Please select assets with more data.')
        return None, None
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    cov_matrix = np.cov(return_vec)
    if np.linalg.matrix_rank(cov_matrix) < n:
        st.warning('Selected assets do not provide enough independent data for optimization. Please select different assets.')
        return None, None
    safe_risk_tolerance = min(max(risk_tolerance, 0.01), 0.99)
    mus = 1 - safe_risk_tolerance
    S = opt.matrix(cov_matrix)
    pbar = opt.matrix(np.mean(return_vec, axis=1))
    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    try:
        portfolios = solvers.qp(mus * S, -pbar, G, h, A, b)
        w = portfolios['x'].T
        Alloc = pd.DataFrame(data=np.array(portfolios['x']), index=assets_selected.columns)
        returns_final = (np.array(assets_selected) * np.array(w))
        returns_sum = np.sum(returns_final, axis=1)
        returns_sum_pd = pd.DataFrame(returns_sum, index=assets.index)
        returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0, :] + 100
        return Alloc, returns_sum_pd
    except Exception as e:
        st.warning(f'Optimization failed: {e}. Please select a different set of assets or adjust your risk tolerance.')
        return None, None

# --- Streamlit UI ---
def main():
    st.markdown("""
        <style>
        .main {background-color: #f8f9fa;}
        .stButton>button {background-color: #4CAF50; color: white; font-weight: bold;}
        .stSidebar {background-color: #f0f2f6;}
        </style>
    """, unsafe_allow_html=True)

    st.title('🤖 Robo Advisor Dashboard')
    with st.expander('ℹ️ How to interpret the results?', expanded=True):
        st.markdown('''
**1. Risk Tolerance (0-1):**  
This value represents your willingness and ability to take investment risk, predicted from your personal and financial information.  
A value close to 0 means you are very risk-averse (prefer safer investments).  
A value close to 1 means you are very risk-tolerant (comfortable with higher risk for higher potential returns).

**2. Asset Allocation - Mean-Variance Allocation:**  
The bar chart shows how your investment would be distributed across the selected assets.  
Higher bars mean a larger portion of your portfolio is allocated to that asset.  
The allocation is optimized to match your risk tolerance: more risk-tolerant users may see more allocation to volatile assets.

**3. Portfolio Value of $100 Investment:**  
The line chart shows how $100 invested in your optimized portfolio would have performed over time, based on historical data.  
A rising line means your portfolio would have grown; a falling line means it would have lost value.  
This helps you visualize the potential risk and reward of your chosen portfolio.

**4. Warnings and Suggestions:**  
If you see warnings about data or optimization, try selecting different assets or adjusting your risk tolerance.  
The dashboard may suggest alternative asset combinations for a more robust portfolio.

**Tip:**  
Use the dashboard to experiment with different profiles and asset selections to find a portfolio that matches your comfort with risk and investment goals. The results are based on historical data and model predictions, so always consider consulting a financial advisor for major investment decisions.
        ''')

    investors = load_investor_data()
    assets = load_asset_data()
    model = load_model()

    with st.sidebar:
        st.header('👤 Investor Profile')
        age = st.slider('Age', int(investors['AGE07'].min()), 70, 25)
        networth = st.slider('NetWorth', -1000000, 3000000, 10000, step=10000)
        income = st.slider('Income', -1000000, 3000000, 100000, step=10000)
        education = st.slider('Education Level (scale of 4)', int(investors['EDCL07'].min()), int(investors['EDCL07'].max()), 2)
        married = st.slider('Married', int(investors['MARRIED07'].min()), int(investors['MARRIED07'].max()), 1)
        kids = st.slider('Kids', int(investors['KIDS07'].min()), int(investors['KIDS07'].max()), 3)
        occupation = st.slider('Occupation', int(investors['OCCAT107'].min()), int(investors['OCCAT107'].max()), 3)
        risk = st.slider('Willingness to take Risk', int(investors['RISK07'].min()), int(investors['RISK07'].max()), 3)

    st.markdown('---')
    st.subheader('Portfolio Simulation')
    with st.form(key='advisor_form'):
        st.markdown('#### 1. Set your profile and select your assets:')
        valid_counts = assets.notna().sum()
        asset_labels = [f"{col} (data: {valid_counts[col]})" for col in assets.columns]
        asset_map = dict(zip(asset_labels, assets.columns))
        default_assets = [f"{a} (data: {valid_counts[a]})" for a in ['GOOGL', 'FB', 'GS', 'MS', 'GE', 'MSFT'] if a in valid_counts]
        selected_labels = st.multiselect('Select the assets for the portfolio:', asset_labels, default=default_assets, key='asset_select')
        selected_assets = [asset_map[l] for l in selected_labels]
        submit_button = st.form_submit_button('🚀 Calculate & Update Portfolio')

    if submit_button:
        X_input = [[age, education, married, kids, occupation, income, risk, networth]]
        risk_tolerance = predict_risk_tolerance(model, X_input)
        st.session_state['risk_tolerance'] = risk_tolerance
        st.session_state['selected_assets'] = selected_assets
        st.success(f'Predicted Risk Tolerance: {risk_tolerance:.2f}')
    else:
        risk_tolerance = st.session_state.get('risk_tolerance', 0.5)
        selected_assets = st.session_state.get('selected_assets', [asset_map[l] for l in default_assets])

    st.info(f"**Risk Tolerance (0-1):** {risk_tolerance:.2f}")

    if selected_assets:
        Alloc, returns_sum_pd = get_asset_allocation(risk_tolerance, selected_assets, assets)
        if Alloc is not None and returns_sum_pd is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('**Asset Allocation - Mean-Variance Allocation**')
                st.plotly_chart(go.Figure([go.Bar(x=Alloc.index, y=Alloc.iloc[:, 0], marker=dict(color='#1f77b4'))]), use_container_width=True)
            with col2:
                st.markdown('**Portfolio value of $100 investment**')
                st.plotly_chart(go.Figure([go.Scatter(x=returns_sum_pd.index, y=returns_sum_pd.iloc[:, 0], marker=dict(color='#ff7f0e'))]), use_container_width=True)
        else:
            st.info('Try adjusting your asset selection or risk tolerance for a feasible portfolio.')

if __name__ == '__main__':
    main()
