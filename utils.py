import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from samplics.estimation import TaylorEstimator
import streamlit as st


def get_vars():
    """
    This function scrapes the NHANES website for the variable names and descriptions
    :return: Pandas DataFrame with the variable names and descriptions
    """
    start_time = time.time()

    url = 'https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component=Laboratory&Cycle='
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    tbody = table.find('tbody')

    data = []

    rows = tbody.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        row_data = [cell.get_text(strip=True) for cell in cells]
        data.append(row_data)

    df = pd.DataFrame(data, columns=['Variable Name', 'Variable Description', 'Data File Name', 'Data File Description',
                                     'Begin Year', 'EndYear',
                                     'Component', 'Use Constraints'])

    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Update complete in: {elapsed_time:.1f} seconds")

    return df


def filter_vars(df_all):
    """
    This function filters the variables to only include the desired variables
    :param df:
    :return: Pandas DataFrame with the filtered variables
    """
    # Filter All Variables
    df_all = df_all[
        ~df_all['Variable Description'].str.contains('sequence number|weight|comment|comt|code', case=False)]

    # Change Variable Types
    df_all['EndYear'] = df_all['EndYear'].astype(int)
    df_all['Begin Year'] = df_all['Begin Year'].astype(int)

    # Only get Variables that start with LBX or URX
    df_all = df_all[df_all['Variable Name'].str.startswith(('LBX', 'URX'))]

    # Find the largest 'Year End' value for each unique 'Variable Name'
    max_years = df_all.groupby('Variable Name')['EndYear'].max()

    # Merge the original DataFrame with the 'max_years' to filter rows
    filtered_df = df_all.merge(max_years, on=['Variable Name', 'EndYear'])

    # Merge on Varaible Name to assign most recent variable description to all past variables.
    df_all = df_all.merge(filtered_df[['Variable Name', 'Variable Description']], on='Variable Name', how='left',
                          suffixes=(' Old', ' New'))

    # Add a Column with Variable Names and Descriptions
    df_all['Full Name'] = df_all['Variable Name'] + ': ' + df_all['Variable Description New']

    # Remove Variables with Use Constraints
    df_all = df_all[df_all['Use Constraints'] == 'None']

    return df_all


def add_yr_demo(df_all):
    """
    This function adds the 'Year' and 'Demo File Name' columns to the DataFrame
    :param df_all:
    :return: Pandas DataFrame with the added columns
    """
    # Add a column with the year range
    df_all['Year'] = df_all.apply(calculate_year, axis=1)

    # Add a column with the demo file name
    df_all['Demo File Name'] = df_all.apply(calculate_demo_file_name, axis=1)

    return df_all


# Custom function to calculate the "Year" column value
def calculate_year(row):
    if row['Data File Name'].startswith('P_'):
        return '2017-2018'
    else:
        return f"{row['Begin Year']}-{row['EndYear']}"


# Custom function to calculate the "Demo File Name" column value
def calculate_demo_file_name(row):
    if row['Data File Name'].startswith('P_'):
        return 'P_DEMO'
    else:
        year_range = row['Year'].split('-')
        start_year = int(year_range[0])
        end_year = int(year_range[1])
        if start_year < 2001:
            return 'DEMO'
        else:
            return f"DEMO_{chr(ord('B') + (start_year - 2001) // 2)}"


def add_columns(df_all):
    """
    This function adds the 'Year' and 'Demo File Name' columns to the DataFrame
    :param df_all:
    :return: Pandas DataFrame with the added columns
    """
    # Add a column with the year range
    df_all['Year'] = df_all.apply(calculate_year, axis=1)

    # Add a column with the demo file name
    df_all['Demo File Name'] = df_all.apply(calculate_demo_file_name, axis=1)

    return df_all


# Helper Functions

# Get list of years variable was measured and not constrained
def create_variable_df(var, df_all):
    df = df_all[df_all['Variable Name'] == var].sort_values(by='EndYear')
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


# Choose the appropriate Weights for the analysis
def choose_weights(data_file, data_df, demo_df):
    weights = ''

    # Count data_df columns that start with 'WT'
    list_data_df_weights = [col for col in data_df.columns if col.startswith('WT')]

    if len(list_data_df_weights) == 0:
        # Handle Pandemic
        if data_file.startswith('P_'):
            weights = 'WTMECPRP'
        else:
            weights = 'WTMEC2YR'
    else:
        weights = list_data_df_weights[0]

    return weights


# Get single year data
def get_single_yr_df(var, year, data_file, demo_file):
    data_url = 'https://wwwn.cdc.gov/Nchs/Nhanes/' + year + '/' + data_file + '.XPT'
    demo_url = 'https://wwwn.cdc.gov/Nchs/Nhanes/' + year + '/' + demo_file + '.XPT'
    data_df = pd.read_sas(data_url)
    demo_df = pd.read_sas(demo_url)
    weights = choose_weights(data_file, data_df, demo_df)
    merged_df = data_df.merge(demo_df, on='SEQN', how='inner')
    merged_df['Weights'] = weights

    # Handle Pandemic
    if data_file.startswith('P_'):
        merged_df['Year'] = '2017-2020 Pre-Pandemic'
    else:
        merged_df['Year'] = year

    # Subset the merged df
    merged_df = merged_df[
        ['SEQN', var, 'Year', 'Weights', weights, 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'SDMVSTRA', 'SDMVPSU']]

    return merged_df


# def get_multi_yr_df(var_df, output):
#     # For each year/data file combination, get the data and stack by year.
#     for i in range(len(var_df)):
#         var = var_df['Variable Name'][i]
#         year = var_df['Year'][i]
#         data_file = var_df['Data File Name'][i]
#         demo_file = var_df['Demo File Name'][i]
#
#         update = f'Downloading data for: {year} {var} {data_file} {demo_file}'
#         print(update)
#         output = output + "\n" + update
#
#         if i == 0:
#             multi_yr_df = get_single_yr_df(var, year, data_file, demo_file)
#         else:
#             single_yr_df = get_single_yr_df(var, year, data_file, demo_file)
#             multi_yr_df = pd.concat([multi_yr_df, single_yr_df])
#     print('Download complete')
#     output = output + "\n" + 'Download complete'
#
#     return multi_yr_df, output

def get_multi_yr_df(var_df, container):
    # For each year/data file combination, get the data and stack by year.
    i = 0
    var = var_df['Variable Name'][i]
    for i in range(len(var_df)):
        var = var_df['Variable Name'][i]
        year = var_df['Year'][i]
        data_file = var_df['Data File Name'][i]
        demo_file = var_df['Demo File Name'][i]

        update = f'{var} {year} {data_file} {demo_file}'
        container.text(update)
        print(update)
        if i == 0:
            multi_yr_df = get_single_yr_df(var, year, data_file, demo_file)
            container.empty()
        else:
            single_yr_df = get_single_yr_df(var, year, data_file, demo_file)
            multi_yr_df = pd.concat([multi_yr_df, single_yr_df])
            container.empty()
    print('Download complete')
    container.empty()

    return multi_yr_df


def get_means(df_all, variable, mean_type, domain, max_value, min_value):
    domain = get_domain(domain)
    df_all = recode_df_domains(df_all)

    if max_value is None:
        pass
    else:
        max_value = float(max_value)

    if min_value is None:
        pass
    else:
        min_value = float(min_value)

    unique_pairs = df_all[['Year', 'Weights']].drop_duplicates().values.tolist()

    for i in range(len(unique_pairs)):
        df_part = df_all[(df_all['Year'] == unique_pairs[i][0]) & (df_all['Weights'] == unique_pairs[i][1])]
        weight = df_part['Weights'][0]
        df_part.reset_index(inplace=True)

        if mean_type == 'Geometric':
            if i == 0:
                mean_part = get_geomean(df_part, variable, weight, domain, max_value, min_value)
                df_means = mean_part
            else:
                mean_part2 = get_geomean(df_part, variable, weight, domain, max_value, min_value)
                df_means = pd.concat([df_means, mean_part2], ignore_index=True)

        else:
            if i == 0:
                mean_part = get_amean(df_part, variable, weight, domain, max_value, min_value)
                df_means = mean_part
            else:
                mean_part2 = get_amean(df_part, variable, weight, domain, max_value, min_value)
                df_means = pd.concat([df_means, mean_part2], ignore_index=True)

    df_means = sort_means(df_means)

    return df_means


def handle_max_min(unweighted_df, variable, max_value, min_value):
    if max_value is not None and min_value is None:
        unweighted_df = unweighted_df[unweighted_df[variable] <= max_value]

    if min_value is not None and max_value is None:
        unweighted_df = unweighted_df[unweighted_df[variable] >= min_value]

    if max_value is not None and min_value is not None:
        unweighted_df = unweighted_df[(unweighted_df[variable] >= min_value) & (unweighted_df[variable] <= max_value)]

    unweighted_df.reset_index(inplace=True)

    return unweighted_df


def get_amean(unweighted_df, variable, weight, domain=None, max_value=None, min_value=None):
    """Computes mean and 95% confidence intervals for single survey period

    Params:
    unweighted_df - unweighted_df for single survey period
    variable - varaible of interest as string (e.g. 'LBXCOT')
    weight - specified sample weights to use (e.g. 'WTMEC2YR')
    domain - optional param for specifiying result split by subgroups

    Returns:
    df of means and 95% confidence intervals for specified arguments
    """

    if max_value is not None or min_value is not None:
        unweighted_df = handle_max_min(unweighted_df, variable, max_value, min_value)

    var_prop = TaylorEstimator("mean")

    if domain == None:
        var_prop.estimate(y=unweighted_df[variable],
                          samp_weight=unweighted_df[weight],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          remove_nan=True)
    else:
        var_prop.estimate(y=unweighted_df[variable],
                          samp_weight=unweighted_df[variable],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          domain=unweighted_df[domain],
                          remove_nan=True)

    df = var_prop.to_dataframe()

    df = format_means(df, unweighted_df, domain, 'Arithmetic')

    return df


def get_geomean(unweighted_df, variable, weight, domain=None, max_value=None, min_value=None):
    """Computes geomean and 95% confidence intervals for single survey period

    Params:
    unweighted_df - unweighted_df for single survey period
    variable - varaible of interest as string (e.g. 'LBXCOT')
    weight - specified sample weights to use (e.g. 'WTMEC2YR')
    domain - optional param for specifiying result split by subgroups

    Returns:
    df of geomeans and 95% confidence intervals for specified arguments
    """

    if max_value is not None or min_value is not None:
        unweighted_df = handle_max_min(unweighted_df, variable, max_value, min_value)

    var_prop = TaylorEstimator("mean")

    if domain == None:
        var_prop.estimate(y=np.log(unweighted_df[variable]),
                          samp_weight=unweighted_df[weight],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          remove_nan=True)

    else:
        var_prop.estimate(y=np.log(unweighted_df[variable]),
                          samp_weight=unweighted_df[weight],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          domain=unweighted_df[domain],
                          remove_nan=True)

    df = var_prop.to_dataframe()
    df['_estimate'] = np.e ** df['_estimate']
    df['_lci'] = np.e ** df['_lci']
    df['_uci'] = np.e ** df['_uci']

    df = format_means(df, unweighted_df, domain, 'Geometric')

    return df


def format_means(df, unweighted_df, domain, mean):
    if domain == None:
        df.rename(columns={"_estimate": "Mean", "_lci": "lower_95%CI", "_uci": "upper_95%CI"}, inplace=True)
        df['Weights'] = unweighted_df['Weights'][0]
        df['Year'] = unweighted_df['Year'][0]
        df['Category'] = 'Total Population'
        df = df[['Category', 'Year', 'Mean', 'lower_95%CI', 'upper_95%CI', 'Weights']]
    else:
        df.rename(columns={"_estimate": "Mean", "_lci": "lower_95%CI", "_uci": "upper_95%CI", "_domain": "Category"},
                  inplace=True)
        df['Weights'] = unweighted_df['Weights'][0]
        df['Year'] = unweighted_df['Year'][0]
        df = df[['Category', 'Year', 'Mean', 'lower_95%CI', 'upper_95%CI', 'Weights']]

    df.loc[:, 'Mean'] = df['Mean'].round(3)
    df.loc[:, 'lower_95%CI'] = df['lower_95%CI'].round(3)
    df.loc[:, 'upper_95%CI'] = df['upper_95%CI'].round(3)

    #     column_names = pd.MultiIndex.from_tuples([('', 'Category'), ('Survey', 'Years'), (mean, 'Mean'),("Lower", "95%CI"), ("Upper", "95%CI"), ('', 'Weights')])
    #     df.columns = column_names

    return df


def get_domain(domain_name):
    if domain_name == 'Race/Hispanic origin':
        domain = 'RIDRETH1'
    elif domain_name == 'Gender':
        domain = 'RIAGENDR'
    elif domain_name == 'Age':
        domain = 1
    elif domain_name == 'Total Population':
        domain = None
    elif domain_name == 'Youth Age Group':
        domain = 'Youth Age Group'
    else:
        print('Error: No domain selected')
    return domain


def recode_df_domains(df):
    # Map number to Race
    race_mapping = {
        1.0: 'Mexican American',
        2.0: 'Other Hispanic',
        3.0: 'Non-Hispanic White',
        4.0: 'Non-Hispanic Black',
        5.0: 'Other Race - Including Multi-Racial'}
    df['RIDRETH1'] = df['RIDRETH1'].replace(race_mapping)

    # Map number to Gender
    gender_mapping = {
        1.0: 'Male',
        2.0: 'Female'}
    df['RIAGENDR'] = df['RIAGENDR'].replace(gender_mapping)

    # Create Youth Age Groups
    conditions = [
        df['RIDAGEYR'].apply(lambda x: 3 <= x < 6),
        df['RIDAGEYR'].apply(lambda x: 6 <= x < 12),
        df['RIDAGEYR'].apply(lambda x: 12 <= x < 20)]

    choices = [
        'a. Age 03-05 years',
        'b. Age 06-11 years',
        'c. Age 12-19 years']

    df['Youth Age Group'] = np.select(conditions, choices, default='d. Age 20+ years')

    return df


def sort_means(df_means):
    df_means = df_means.sort_values(by=['Category', 'Year'])
    #     df_means = df_means.sort_values(by=[('', 'Category'), ('Survey','Years')])
    df_means.reset_index(drop=True, inplace=True)

    return df_means
