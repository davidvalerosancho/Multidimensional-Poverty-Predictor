import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting page layout to full width

# Title
st.title('Income and living conditions in Spain')

# Description
st.markdown('''
Find out about Spain's income and living conditions for its adult population!. This app helps you perform interactive visualizations on different variables.\n
The data has been retrieved from the 2019 Income and Living Conditions survey conducted by the INE
(*Instituto Nacional de Estadistica*), Spain's official agency for statistical services.\n
This survey aims at collecting information on income, poverty, social exclusion and living conditions.\n
* **Python libraries**: streamlit, pandas, matplotlib, seaborn
* **Data Source**: [ine.es](https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176807&menu=ultiDatos&idp=1254735976608)
''')

#Loading data from Git Repo
url = 'https://raw.githubusercontent.com/deividvalerius/Wealth-Estimator/master/Data/all_variables.csv'
df = pd.read_csv(url)

#Creating sidebar for interactive user imputation
st.sidebar.header('User Input Features')

#Sidebar - Sex
sex = sorted(df.sex.unique())
selected_sex = st.sidebar.multiselect('Sex', sex, sex)

#Sidebar - Civil status
civil_status = sorted(df.civil_status.unique())
selected_civil_status = st.sidebar.multiselect('Civil status', civil_status, civil_status)

#Sidebar - Familial status
familial_status = sorted(df.familial_status.unique())
selected_familial_status = st.sidebar.multiselect('Familial status', familial_status, familial_status)

#Sidebar - Region
region = sorted(df.region.unique())
selected_region = st.sidebar.multiselect('Region', region, region)

#Sidebar - Population density
population_density = sorted(df.population_density.unique())
selected_population_density = st.sidebar.multiselect('Population density', population_density, population_density)

#Sidebar - Citizenship
citizenship = sorted(df.citizenship.unique())
selected_citizenship = st.sidebar.multiselect('Citizenship', citizenship, citizenship)

#Sidebar - Tenure status
tenure_status = sorted(df.tenure_status.unique())
selected_tenure_status = st.sidebar.multiselect('Tenure status', tenure_status, tenure_status)

#Sidebar - Education level
education_level = sorted(df.education_level.unique())
selected_education_level = st.sidebar.multiselect('Education level', education_level, education_level)

#Sidebar - Working status
working_status = sorted(df.working_status.unique())
selected_working_status = st.sidebar.multiselect('Working status', working_status, working_status)

#Sidebar - Occupation
occupation = sorted(df.occupation.unique())
selected_occupation = st.sidebar.multiselect('Occupation', occupation, occupation)

#Sidebar - Bad health
bad_health = sorted(df.bad_health.unique())
selected_bad_health = st.sidebar.multiselect('Bad health', bad_health, bad_health)

#Sidebar - Material deprivation
material_deprivation = sorted(df.material_deprivation.unique())
selected_material_deprivation = st.sidebar.multiselect('Material deprivation', material_deprivation, material_deprivation)

#Sidebar - Age
selected_age = st.sidebar.slider('Age', int(df.age.min()), int(df.age.max()), (int(df.age.min()), int(df.age.max())))

#Sidebar - Years worked
selected_years_worked = st.sidebar.slider('Years worked', int(df.years_worked.min()), int(df.years_worked.max()), (int(df.years_worked.min()), int(df.years_worked.max())))

#Sidebar - Hours a week worked
selected_hours_week_worked = st.sidebar.slider('Hours a week worked', int(df.hours_week_worked.min()), int(df.hours_week_worked.max()), (int(df.hours_week_worked.min()), int(df.hours_week_worked.max())))

#Sidebar - Adjusted income
selected_adjusted_income = st.sidebar.slider('Adjusted income', -20000, 180000, (-20000, 180000), 1000)

#Showing selected data frame
selected_df = df[df.sex.isin(selected_sex)
            & df.civil_status.isin(selected_civil_status)
            & df.familial_status.isin(selected_familial_status)
            & df.region.isin(selected_region)
            & df.population_density.isin(selected_population_density)
            & df.citizenship.isin(selected_citizenship)
            & df.tenure_status.isin(selected_tenure_status)
            & df.education_level.isin(selected_education_level)
            & df.working_status.isin(selected_working_status)
            & df.occupation.isin(selected_occupation)
            & df.material_deprivation.isin(selected_material_deprivation)].reset_index()

#Subtitle for the variable options
st.subheader('Choose a variable for display')

#Selectbox for the X and Y variable
variable_names = {c[0].upper() + c[1:].lower().replace('_', ' '): c for c in df.columns} #editing string format from the df column names
x = st.selectbox('X', tuple(variable_names.keys())[1:])
y = st.selectbox('Y', tuple(variable_names.keys())[1:])

#Only X variable option
one_variable_button = st.button('Select only X variable')

#Defining functions
def weighted_freq(data, cat_column):
    dummy = pd.get_dummies(cat_column)
    for c in dummy.columns:
        dummy['weight_' + str(c)] = dummy[c] * data.weight
    freq_dict = {}
    for c in dummy.columns:
        if str(c)[0] == 'w':
            freq_dict[c[7:]] = dummy[c].sum() / data.weight.sum()
    return freq_dict

def barplot(x=variable_names[x], data=selected_df, legend=True):
    fig, ax = plt.subplots()
    labels = sorted(data[x].unique())
    label_colors = ['C'+str(i) for i in range(len(labels))]
    x_ticks = range(len(labels))
    freq = weighted_freq(data, data[x])
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    ax = plt.subplot()
    ax.bar(x_ticks, [freq[label] for label in labels], color=label_colors)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)
    plt.xlabel(x)
    plt.ylabel('Frequency')
    if legend==False:
        ax.set_xticklabels(labels)
    else:
        ax.set_xticklabels(['' for label in labels])
        for i in range(len(labels)):
            ax.bar([0], [0], label=labels[i][:40], color=label_colors[i])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

def histplot(x=variable_names[x], data=selected_df, bins=68):
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    ax = sns.histplot(x=data[x], data=data, weights='weight', bins=bins, stat='density')
    st.pyplot(fig)

def length_calculator(labels):
    a = 4
    multiplier = 1
    rows = 2
    for i in range(labels):
        if i+1 >= a + 3:
            a += 3
            multiplier += 0.5
            rows += 1
    return multiplier, rows

def bardiagram(x=variable_names[x], hue=variable_names[y], data=selected_df, legend=True):
    fig, ax = plt.subplots()
    titles = sorted(data[x].unique())
    labels = sorted(data[hue].unique())
    label_colors = ['C'+str(i) for i in range(len(labels))]
    x_ticks = range(len(labels))
    length_multiplier, subplot_rows = length_calculator(len(titles))
    fig = plt.figure(figsize=(6.4*2, 4.8*2*length_multiplier))
    subplot = 1
    for title in titles:
        subplot_data = data[data[x] == title]
        freq = weighted_freq(subplot_data, subplot_data[hue])
        ax = plt.subplot(subplot_rows, 3, subplot)
        ax.title.set_text(title[:40])
        ax.bar(x_ticks, [freq[label] if label in freq.keys() else 0 for label in labels], color=label_colors)
        ax.set_xticks(x_ticks)
        if legend==False:
            ax.set_xticklabels(labels)
        else:
            ax.set_xticklabels(['' for label in labels])
        subplot += 1
    if legend==True:
        for i in range(len(labels)):
            ax.bar([0], [0], label=labels[i][:20], color=label_colors[i])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

def weighted_cat(data, cat_column):
    dummy = pd.get_dummies(cat_column)
    for c in dummy.columns:
        dummy['weight_' + str(c)] = dummy[c] * data.weight
    weighted_cat = []
    for i in range(len(data)):
        for c in dummy.columns:
            if c[0] == 'w':
                for x in range(int(round(dummy[c][i]/1000, 0))):
                       weighted_cat.append(c[7:])
    return weighted_cat

def weighted_num(data, num_column):
    weighted_num = []
    for i in range(len(data)):
        for x in range(int(round(data.weight[i]/1000, 0))):
            weighted_num.append(num_column[i])
    return weighted_num

def boxplot(x=variable_names[x], y=variable_names[y], data=selected_df, legend=True):
    fig, ax = plt.subplots()
    weighted_data = pd.DataFrame({x: weighted_cat(data, data[x]), y: weighted_num(data, data[y])})
    labels = sorted(data[x].unique())
    palette = {labels[i]: 'C'+str(i) for i in range(len(labels))}
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    ax = plt.subplot()
    boxplot = sns.boxplot(x=x, y=y, data=weighted_data, order=sorted(data[x].unique()), palette=palette)
    if legend == True:
        boxplot.set(xticklabels=[])
        for i in range(len(labels)):
            ax.bar([0], [0], label=labels[i][:40], color=palette[labels[i]])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

def scatterplot(x=variable_names[x], y=variable_names[y], data=selected_df):
    fig, ax = plt.subplots()
    weighted_data = pd.DataFrame({x: weighted_num(data, data[x]), y: weighted_num(data, data[y])})
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    ax = sns.scatterplot(x=x, y=y, data=weighted_data)
    st.pyplot(fig)

#Showing graph
if one_variable_button == False:
    if str(selected_df[variable_names[x]].dtype) == 'object':
        if str(selected_df[variable_names[y]].dtype) == 'object':
            if st.button('Hide legend'):
                bardiagram(legend=False)
            else:
                bardiagram()
        else:
            if st.button('Hide legend'):
                boxplot(legend=False)
            else:
                boxplot()
    else:
        if str(selected_df[variable_names[y]].dtype) == 'object':
            st.write('Swap the X and Y variables to get a boxplot!')
        else:
            scatterplot()

else:
    if str(selected_df[variable_names[x]].dtype) == 'object':
        if st.button('Hide legend'):
            barplot(legend=False)
        else:
            barplot()
    else:
        bins = st.slider('bins', 1, 100, 68)
        histplot(bins=bins)

st.subheader('Check out our predictive model!')
