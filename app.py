import streamlit as st
import utils
from utils import *
import time

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'selected_var' not in st.session_state:
    st.session_state.selected_var = None
if 'multi_yr_df' not in st.session_state:
    st.session_state.multi_yr_df = None
if 'selected_variable' not in st.session_state:
    st.session_state.selected_variable = None

st.title('NHANES Weighted Data Explorer')

# Create button for updating available variables
if st.button('Update available variables'):
    start = time.time()
    # Display a message while the data is being updated
    with st.spinner('Updating data...'):
        df = utils.get_vars()
        variables = utils.filter_vars(df)
        variables = utils.add_yr_demo(variables)
        variables.to_csv(r'temp_data\var_list.csv', index=False)
        end = time.time()
        st.success(f'Data updated in {round(end - start, 2)} seconds!')

# Load the data
variables = pd.read_csv(r'temp_data\var_list.csv')

# Variable selection
unique_variables = sorted(variables['Full Name'].unique())
selected_variable = st.selectbox('Select a variable', unique_variables, key='variable_selector')

# Create a 'Download Data' button and on click, display the next set of options
if st.button('Download Data'):
    st.session_state.data_loaded = False

    var = selected_variable.split(':')[0].strip()
    var_df = utils.create_variable_df(var, variables)
    start = time.time()
    with st.spinner('Downloading data from NHANES for...'):
        container = st.empty()
        multi_yr_df = utils.get_multi_yr_df(var_df, container)
        multi_yr_df.to_csv(r'temp_data\multi_yr_df.csv', index=False)
        end = time.time()
        st.success(f'Download complete in {round(end - start, 2)} seconds!')
        st.session_state.data_loaded = True
        st.session_state.selected_var = var
        st.session_state.multi_yr_df = multi_yr_df
        st.session_state.selected_variable = selected_variable

if st.session_state.selected_variable != selected_variable:
    st.session_state.data_loaded = False

if st.session_state.data_loaded:
    # Load the data
    # multi_yr_df = pd.read_csv(r'temp_data\multi_yr_df.csv')
    multi_yr_df = st.session_state.multi_yr_df
    var = st.session_state.selected_var

    # Analysis selection
    analysis_options = ['Summary Statistics', 'Histograms', 'Graphs']
    selected_analysis = st.selectbox('Select an analysis', analysis_options)

    # Make second dropdown choices
    mean_types = ['Geometric', 'Arithmetic']
    domains = ['Total Population', 'Gender', 'Race/Hispanic origin', 'Youth Age Group']

    # Create a column layout for the Min and Max text inputs
    col1, col2 = st.columns(2)
    with col1:
        selected_mean = st.selectbox('Select a mean type', mean_types)
        selected_min = st.text_input('Min', value=None)
    with col2:
        selected_domain = st.selectbox('Select a domain', domains)
        selected_max = st.text_input('Max', value=None)

    # Create a 'Submit' button and on click, display the results
    if st.button('Submit'):
        try:
            if selected_analysis == 'Summary Statistics':
                df_means = utils.get_means(multi_yr_df,
                                           var,
                                           selected_mean,
                                           domain=selected_domain,
                                           max_value=selected_max,
                                           min_value=selected_min)
                st.write(f'{selected_analysis} for {var}')
                st.write(df_means)
            elif selected_analysis == 'Histograms':
                st.write('Histograms')
            elif selected_analysis == 'Graphs':
                st.write('Graphs')
            else:
                st.write('No analysis selected')
        except Exception as e:
            st.error(f"An error occurred: {e}")
