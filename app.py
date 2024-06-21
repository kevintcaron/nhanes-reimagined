import streamlit as st
import utils
from utils import *
import time
import plotly.express as px
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# page configuration
st.set_page_config(
    page_title="NHANES Reimagined",
    layout='wide'
)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# authenticator.login('Login', 'main')
colX, colY, colZ  = st.columns([2, 3, 2])
with colY:
    authenticator.login()

name = st.session_state['name']
authentication_status = st.session_state['authentication_status']
username = st.session_state['username']

# hashed_password = stauth.Hasher(['example']).generate()
# st.write(hashed_password)
containerTop = st.container()
st.markdown('#')
containerBottom = st.container()

if st.session_state["authentication_status"]:
    with containerTop:
        colA, colB, colC  = st.columns([2, 4, 2])
        with colB:
            authenticator.logout('Logout', 'main', key='unique_key')
            st.write(f'Welcome *{st.session_state["name"]}*')

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
            st.write('To provide feedback on this tool, please use our [Feedback Form](https://forms.office.com/g/2mqJRn0Paq)')


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
                multi_yr_df = st.session_state.multi_yr_df
                var = st.session_state.selected_var
                easy_name = st.session_state.selected_variable.split(':')[1].strip()

                # Analysis selection
                analysis_options = ['Summary Statistics', 'Compare Histograms', 'Line Graph']
                selected_analysis = st.selectbox('Select an analysis', analysis_options)

                # Make second dropdown choices
                mean_types = ['Geometric', 'Arithmetic']
                domains = ['Total Population', 'Gender', 'Race/Hispanic origin', 'Youth Age Group']

                # Create a column layout for the Min and Max text inputs
                col1, col2 = st.columns(2)
                with col1:
                    selected_mean = st.selectbox('Select a mean type', mean_types)
                    selected_min = st.text_input('Min', value=None)
                    if selected_analysis == 'Compare Histograms':
                        selected_survey = st.selectbox('Select a survey year', multi_yr_df['Year'].unique())

                with col2:
                    selected_domain = st.selectbox('Select a demographic category', domains)
                    selected_max = st.text_input('Max', value=None)
                    if selected_analysis == 'Compare Histograms':
                        selected_bins = st.text_input('Number of bins', value=40)
                        log = st.checkbox('Log scale')

                

        # Create a 'Submit' button and on click, display the results
        with colB:
            if st.session_state.data_loaded:
                if st.button('Submit'):
                    try:
                        if selected_analysis == 'Summary Statistics':
                            df_means = utils.get_means(multi_yr_df,
                                                    var,
                                                    selected_mean,
                                                    domain=selected_domain,
                                                    max_value=selected_max,
                                                    min_value=selected_min,
                                                    purpose='summary statistics')
                            containerBottom.write(
                                f'{easy_name} {selected_mean} Means by {selected_domain} (Min: {selected_min}, Max: {selected_max})')

                            # dynamically generate table height to remove scrolling
                            df_height = (len(df_means) + 1) * 35 + 3
                            containerBottom.dataframe(df_means, hide_index=True, use_container_width=True, height=df_height)

                            # display general footer notes
                            containerBottom.write("Please note 2017-2020 Pre-Pandemic values might not be comparable to other survey periods and are \
                                                  displayed soley for hypothesis generation and completion.")

                            # display dynamic footer notes
                            if '*' in set(df_means[f'{selected_mean} Mean']):
                                containerBottom.write(r"\* Not calculated: Proportion of results below limit of detection was too high to provide a valid result.")

                        elif selected_analysis == 'Compare Histograms':
                            with st.spinner('Creating Histograms...'):
                                utils.compare_frequency(multi_yr_df,
                                                        easy_name,
                                                        selected_survey,
                                                        var,
                                                        selected_domain,
                                                        selected_bins,
                                                        log,
                                                        selected_min,
                                                        selected_max)
                        elif selected_analysis == 'Line Graph':
                            df_means = utils.get_means(multi_yr_df,
                                                    var,
                                                    selected_mean,
                                                    domain=selected_domain,
                                                    max_value=selected_max,
                                                    min_value=selected_min,
                                                    purpose='line graph')

                            # Create graph of df_means with Year as x-axis and Mean as y-axis
                            fig = px.line(df_means, x='Year', y=f'{selected_mean} Mean', color='Category')
                            # Add title and axis labels
                            fig.update_layout(title=f'{easy_name} {selected_mean} Means by {selected_domain}',
                                            xaxis_title='Year',
                                            yaxis_title=f'{easy_name}')
                            # Add upper and lower 95% confidence intervals as whiskers
                            fig.update_traces(error_y=dict(type='data', array=df_means['upper_95%CI'] - df_means[f'{selected_mean} Mean'],
                                                        arrayminus=df_means[f'{selected_mean} Mean'] - df_means['lower_95%CI']))
                            # Add upper and lower 95% confidence intervals as error bars
                            containerBottom.plotly_chart(fig)
                        else:
                            st.write('No analysis selected')
                    except Exception as e:
                        st.error(f"An error occurred: {e}")


    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #f1f1f1;
                text-align: center;
                padding: 10px;
                font-size: 12px;
                color: black;
            }
        </style>
        <div class="footer">
            Created by Kevin Caron, Steve Arnstein, and Aaron Adams
        </div>
        """,
        unsafe_allow_html=True
    )

elif st.session_state["authentication_status"] is False:
    with colY:
        st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    with colY:
        st.warning('Please enter your username and password')
