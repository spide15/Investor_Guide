import streamlit as st
import sys
import sklearn.ensemble._forest
import sklearn.tree._tree
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble._forest
sys.modules['sklearn.tree.tree'] = sklearn.tree._tree
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pickle import load
import cvxopt as opt
from cvxopt import blas, solvers

# Load data
def load_data():
    investors = pd.read_csv('InputData.csv', index_col=0)
    assets = pd.read_csv('SP500Data.csv', index_col=0)
    missing_fractions = assets.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    assets.drop(labels=drop_list, axis=1, inplace=True)
    assets = assets.ffill()
    return investors, assets

def predict_riskTolerance(X_input):
    filename = 'finalized_model.sav'
    try:
        loaded_model = load(open(filename, 'rb'))
    except Exception as e:
        # If loading fails, retrain and save a new model using available data
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from pickle import dump
        # Try to load training data from InputData.csv (must match training features)
        investors = pd.read_csv('InputData.csv', index_col=0)
        # The following columns must match the original training features
        feature_cols = ['AGE07','EDCL07','MARRIED07','KIDS07','OCCAT107','INCOME07','RISK07','NETWORTH07']
        target_col = 'TrueRiskTol'
        if target_col in investors.columns:
            X_train = investors[feature_cols]
            y_train = investors[target_col]
            model = RandomForestRegressor(n_estimators=250)
            model.fit(X_train, y_train)
            dump(model, open(filename, 'wb'))
            loaded_model = model
        else:
            raise RuntimeError('Model file is incompatible and no training data with target available to retrain.')
    predictions = loaded_model.predict(X_input)
    return predictions

def get_asset_allocation(riskTolerance, stock_ticker, assets):
    assets_selected = assets.loc[:, stock_ticker]
    if len(stock_ticker) < 2:
        st.warning('Please select at least 2 assets for portfolio optimization.')
        return None, None
    # Data check: show number of valid data points for each asset
    valid_counts = assets_selected.notna().sum()
    min_valid = valid_counts.min()
    st.info('Valid data points per asset:')
    st.write(valid_counts)
    if min_valid < 10:
        st.warning('One or more selected assets have too few valid data points (<10). Please select assets with more data.')
        return None, None
    # Check for highly correlated assets
    corr_matrix = assets_selected.corr().abs()
    high_corr = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) > 0.98).any().any()
    if high_corr:
        st.warning('Some selected assets are highly correlated (>0.98). Consider diversifying your selection for a more robust portfolio.')
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    # Check if covariance matrix is full rank
    cov_matrix = np.cov(return_vec)
    if np.linalg.matrix_rank(cov_matrix) < n:
        st.warning('Selected assets do not provide enough independent data for optimization. Please select different assets.')
        # Suggest a random valid portfolio
        st.info('Suggesting a random valid portfolio:')
        valid_assets = valid_counts[valid_counts >= 10].index.tolist()
        if len(valid_assets) >= 2:
            import random
            suggestion = random.sample(valid_assets, min(6, len(valid_assets)))
            st.write('Try this combination:', suggestion)
        return None, None
    # Clamp riskTolerance to avoid infeasible optimization
    safe_risk_tolerance = min(max(riskTolerance, 0.01), 0.99)
    if safe_risk_tolerance != riskTolerance:
        st.warning(f'Risk Tolerance value was clamped from {riskTolerance:.4f} to {safe_risk_tolerance:.2f} to ensure optimization stability. Consider adjusting your risk profile for more practical results.')
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
    except ValueError as e:
        st.warning(f'Optimization failed: {e}. Please select a different set of assets or adjust your risk tolerance.')
        # Suggest a random valid portfolio
        st.info('Suggesting a random valid portfolio:')
        valid_assets = valid_counts[valid_counts >= 10].index.tolist()
        if len(valid_assets) >= 2:
            import random
            suggestion = random.sample(valid_assets, min(6, len(valid_assets)))
            st.write('Try this combination:', suggestion)
        return None, None

def main():
    st.title('Robo Advisor Dashboard')
    with st.expander('How to interpret the results?', expanded=True):
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
    investors, assets = load_data()
    st.sidebar.header('Investor Characteristics')
    # User input for investor profile
    age = st.sidebar.slider('Age', int(investors['AGE07'].min()), 70, 25)
    networth = st.sidebar.slider('NetWorth', -1000000, 3000000, 10000)
    income = st.sidebar.slider('Income', -1000000, 3000000, 100000)
    education = st.sidebar.slider('Education Level (scale of 4)', int(investors['EDCL07'].min()), int(investors['EDCL07'].max()), 2)
    married = st.sidebar.slider('Married', int(investors['MARRIED07'].min()), int(investors['MARRIED07'].max()), 1)
    kids = st.sidebar.slider('Kids', int(investors['KIDS07'].min()), int(investors['KIDS07'].max()), 3)
    occupation = st.sidebar.slider('Occupation', int(investors['OCCAT107'].min()), int(investors['OCCAT107'].max()), 3)
    risk = st.sidebar.slider('Willingness to take Risk', int(investors['RISK07'].min()), int(investors['RISK07'].max()), 3)

    # Calculate risk tolerance
    if st.sidebar.button('Calculate Risk Tolerance'):
        X_input = [[age, education, married, kids, occupation, income, risk, networth]]
        risk_tolerance = predict_riskTolerance(X_input)
        # Clamp and store as float between 0 and 1
        risk_tolerance = float(risk_tolerance)
        st.session_state['risk_tolerance'] = risk_tolerance
        st.success(f'Predicted Risk Tolerance: {risk_tolerance:.2f}')
    risk_tolerance = st.session_state.get('risk_tolerance', 0.5)
    st.write(f"Risk Tolerance (0-1): {risk_tolerance:.2f}")

    st.header('Asset Allocation and Portfolio Performance')
    options = list(assets.columns)
    # Asset selection with data quality info
    valid_counts = assets.notna().sum()
    asset_labels = [f"{col} (data: {valid_counts[col]})" for col in options]
    asset_map = dict(zip(asset_labels, options))
    selected_labels = st.multiselect('Select the assets for the portfolio:', asset_labels, default=[f"{a} (data: {valid_counts[a]})" for a in ['GOOGL', 'FB', 'GS', 'MS', 'GE', 'MSFT'] if a in valid_counts])
    selected_assets = [asset_map[l] for l in selected_labels]

    if st.button('Submit') and selected_assets:
        Alloc, returns_sum_pd = get_asset_allocation(risk_tolerance, selected_assets, assets)
        if Alloc is not None and returns_sum_pd is not None:
            st.subheader('Asset Allocation - Mean-Variance Allocation')
            st.plotly_chart(go.Figure([go.Bar(x=Alloc.index, y=Alloc.iloc[:, 0], marker=dict(color='red'))]))
            st.subheader('Portfolio value of $100 investment')
            st.plotly_chart(go.Figure([go.Scatter(x=returns_sum_pd.index, y=returns_sum_pd.iloc[:, 0], marker=dict(color='red'))]))
        else:
            st.info('Try adjusting your asset selection or risk tolerance for a feasible portfolio.')

if __name__ == '__main__':
    main()
